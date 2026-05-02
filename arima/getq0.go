package arima

import "gonum.org/v1/gonum/mat"

// stationaryCovGardner computes the stationary covariance of the ARMA(p,q)
// state-space form using Gardner's algorithm — a numerically-stable
// alternative to direct discrete-Lyapunov solution.
//
// Mirrors R's `stats::C_getQ0` (in src/library/stats/src/arima.c). Used during
// the exact diffuse Kalman filter to seed the ARMA block of P_*.
//
// phi: non-seasonal AR coefficients (length p, fully expanded if seasonal)
// theta: non-seasonal MA coefficients (length q, fully expanded if seasonal)
// Returns an r×r mat.Dense where r = max(p, q+1).
func stationaryCovGardner(phi, theta []float64) *mat.Dense {
	p := len(phi)
	q := len(theta)
	r := p
	if q+1 > r {
		r = q + 1
	}
	if r == 0 {
		return mat.NewDense(0, 0, nil)
	}
	np := r * (r + 1) / 2
	nrbar := np * (np - 1) / 2
	if nrbar < 0 {
		nrbar = 0
	}

	// V[ind] = vi * vj where vk = (k == 0 ? 1 : (k-1 < q ? theta[k-1] : 0))
	V := make([]float64, np)
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

	P := make([]float64, r*r) // final r×r matrix, flat column-major like R

	if r == 1 {
		if p == 0 {
			P[0] = 1
		} else {
			P[0] = 1 / (1 - phi[0]*phi[0])
		}
		out := mat.NewDense(1, 1, []float64{P[0]})
		return out
	}

	if p > 0 {
		// Givens-rotation based solver via inclu2.
		xnext := make([]float64, np)
		xrow := make([]float64, np)
		rbar := make([]float64, nrbar)
		thetab := make([]float64, np)
		Pbuf := make([]float64, np)

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
		Pbuf := make([]float64, np)
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

	// Convert column-major P to a row-major mat.Dense.
	out := mat.NewDense(r, r, nil)
	for i := 0; i < r; i++ {
		for j := 0; j < r; j++ {
			// P (column-major) at row i, col j: P[i + r*j]
			out.Set(i, j, P[i+r*j])
		}
	}
	return out
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
