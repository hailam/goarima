package arima

// PublicGardner exposes Gardner's stationary cov for parity diagnostics.
func PublicGardner(phi, theta []float64) [][]float64 {
	flat, r := stationaryCovGardner(phi, theta)
	out := make([][]float64, r)
	for i := 0; i < r; i++ {
		row := make([]float64, r)
		copy(row, flat[i*r:(i+1)*r])
		out[i] = row
	}
	return out
}

// stationaryCovGardner computes the stationary covariance of the ARMA(p,q)
// state-space form using Gardner's algorithm — a numerically-stable
// alternative to direct discrete-Lyapunov solution.
//
// Mirrors R's `stats::C_getQ0` (in src/library/stats/src/arima.c). Used during
// the exact diffuse Kalman filter to seed the ARMA block of P_*.
//
// phi: non-seasonal AR coefficients (length p, fully expanded if seasonal)
// theta: non-seasonal MA coefficients (length q, fully expanded if seasonal)
// Returns the flat row-major r×r covariance and r itself, where r = max(p, q+1).
// The returned slice has length r*r; the caller can copy or alias as needed.
func stationaryCovGardner(phi, theta []float64) ([]float64, int) {
	var ws gardnerWorkspace
	return stationaryCovGardnerInto(&ws, phi, theta)
}

// stationaryCovGardnerInto is the buffer-reuse variant of
// stationaryCovGardner. All ~7 internal buffers come from `ws` instead
// of being allocated per call. The returned []float64 aliases ws.P,
// valid until the next call on the same workspace. Pre-pool, this
// function was the single largest allocator in AutoArima
// (~36% of total per-Fit allocations).
func stationaryCovGardnerInto(ws *gardnerWorkspace, phi, theta []float64) ([]float64, int) {
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		return nil, 0
	}
	np := r * (r + 1) / 2
	nrbar := np * (np - 1) / 2
	if nrbar < 0 {
		nrbar = 0
	}

	// V[ind] = vi * vj where vk = (k == 0 ? 1 : (k-1 < q ? theta[k-1] : 0))
	ws.V = ensureLen(ws.V, np)
	V := ws.V
	{
		ind := 0
		for j := 0; j < r; j++ {
			vj := 0.0
			if j == 0 {
				vj = 1
			} else if j-1 < q {
				vj = theta[j-1]
			}
			for i := j; i < r; i++ {
				vi := 0.0
				if i == 0 {
					vi = 1
				} else if i-1 < q {
					vi = theta[i-1]
				}
				V[ind] = vi * vj
				ind++
			}
		}
	}

	ws.P = ensureLenZ(ws.P, r*r) // zeroed because we write only some cells
	P := ws.P                    // final r×r matrix, flat column-major like R

	if r == 1 {
		if p == 0 {
			P[0] = 1
		} else {
			P[0] = 1 / (1 - phi[0]*phi[0])
		}
		return P, 1
	}

	if p > 0 {
		// Givens-rotation based solver via inclu2.
		ws.xnext = ensureLenZ(ws.xnext, np)
		ws.xrow = ensureLenZ(ws.xrow, np)
		ws.rbar = ensureLenZ(ws.rbar, nrbar)
		ws.thetab = ensureLenZ(ws.thetab, np)
		ws.Pbuf = ensureLenZ(ws.Pbuf, np)
		xnext := ws.xnext
		xrow := ws.xrow
		rbar := ws.rbar
		thetab := ws.thetab
		Pbuf := ws.Pbuf

		ind := 0
		ind1 := -1
		npr := np - r
		npr1 := npr + 1
		indj := npr
		ind2 := npr - 1

		for j := 0; j < r; j++ {
			phij := 0.0
			if j < p {
				phij = phi[j]
			}
			xnext[indj] = 0
			indj++
			indi := npr1 + j
			for i := j; i < r; i++ {
				ynext := V[ind]
				ind++
				phii := 0.0
				if i < p {
					phii = phi[i]
				}
				if j != r-1 {
					xnext[indj] = -phii
					if i != r-1 {
						xnext[indi] -= phij
						ind1++
						xnext[ind1] = -1
					}
				}
				xnext[npr] = -phii * phij
				ind2++
				if ind2 >= np {
					ind2 = 0
				}
				xnext[ind2] += 1
				inclu2(np, xnext, xrow, ynext, Pbuf, rbar, thetab)
				xnext[ind2] = 0
				if i != r-1 {
					xnext[indi] = 0
					indi++
					xnext[ind1] = 0
				}
			}
		}

		// Backsubstitution
		ithisr := nrbar - 1
		im := np - 1
		for i := 0; i < np; i++ {
			bi := thetab[im]
			jm := np - 1
			for j := 0; j < i; j++ {
				bi -= rbar[ithisr] * Pbuf[jm]
				ithisr--
				jm--
			}
			Pbuf[im] = bi
			im--
		}

		// Re-order P (R's "now re-order p")
		// Take last r entries of Pbuf and store in xnext
		ind = npr
		for i := 0; i < r; i++ {
			xnext[i] = Pbuf[ind]
			ind++
		}
		// Shift first npr entries of Pbuf to the end
		ind = np - 1
		ind1 = npr - 1
		for i := 0; i < npr; i++ {
			Pbuf[ind] = Pbuf[ind1]
			ind--
			ind1--
		}
		for i := 0; i < r; i++ {
			Pbuf[i] = xnext[i]
		}

		// Unpack Pbuf into P (full matrix). R's unpack:
		//   for(i = r-1, ind = np; i > 0; i--)
		//       for(j = r-1; j >= i; j--) P[r*i + j] = P[--ind];
		//   for(i = 0; i < r-1; i++) for(j = i+1; j < r; j++)
		//       P[i + r*j] = P[j + r*i];
		// Here P is treated as column-major flat: P[r*i + j] = P[j, i].
		// We'll work in the same indexing then return as row-major.
		ind = np
		for i := r - 1; i > 0; i-- {
			for j := r - 1; j >= i; j-- {
				ind--
				P[r*i+j] = Pbuf[ind]
			}
		}
		// First column (i=0) stays as Pbuf[0..r-1]
		for j := 0; j < r; j++ {
			P[r*0+j] = Pbuf[j]
		}
		// Symmetrize: P[i, j] = P[j, i] for i < j (column-major: P[i+r*j] = P[j+r*i])
		for i := 0; i < r-1; i++ {
			for j := i + 1; j < r; j++ {
				P[i+r*j] = P[j+r*i]
			}
		}
	} else {
		// Pure MA: backsubstitution with V
		ws.Pbuf = ensureLenZ(ws.Pbuf, np)
		Pbuf := ws.Pbuf
		indn := np
		ind := np
		for i := 0; i < r; i++ {
			for j := 0; j <= i; j++ {
				ind--
				Pbuf[ind] = V[ind]
				if j != 0 {
					indn--
					Pbuf[ind] += Pbuf[indn]
				}
			}
		}
		// Unpack
		ind = np
		for i := r - 1; i > 0; i-- {
			for j := r - 1; j >= i; j-- {
				ind--
				P[r*i+j] = Pbuf[ind]
			}
		}
		for j := 0; j < r; j++ {
			P[r*0+j] = Pbuf[j]
		}
		for i := 0; i < r-1; i++ {
			for j := i + 1; j < r; j++ {
				P[i+r*j] = P[j+r*i]
			}
		}
	}

	// Convert column-major P to row-major and return flat.
	out := make([]float64, r*r)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			// P (column-major) at row i, col j: P[i + r*j]
			out[i*r+j] = P[i+r*j]
		}
	}
	return out, r
}

// inclu2 is the Givens-rotation update used by stationaryCovGardner.
//
// Mirrors stats::inclu2 in arima.c.
func inclu2(np int, xnext, xrow []float64, ynext float64, d, rbar, thetab []float64) {
	for i := 0; i < np; i++ {
		xrow[i] = xnext[i]
	}
	ithisr := 0
	for i := 0; i < np; i++ {
		if xrow[i] != 0 {
			xi := xrow[i]
			di := d[i]
			dpi := di + xi*xi
			d[i] = dpi
			cbar := di / dpi
			sbar := xi / dpi
			for k := i + 1; k < np; k++ {
				xk := xrow[k]
				rbthis := rbar[ithisr]
				xrow[k] = xk - xi*rbthis
				rbar[ithisr] = cbar*rbthis + sbar*xk
				ithisr++
			}
			xk := ynext
			ynext = xk - xi*thetab[i]
			thetab[i] = cbar*thetab[i] + sbar*xk
			if di == 0 {
				return
			}
		} else {
			ithisr += np - i - 1
		}
	}
}
