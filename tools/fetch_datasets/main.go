// fetch_datasets populates the on-demand R-version dataset variants
// (LoadAusbeerR, LoadGasolineForecastR) by downloading from the canonical
// publishers and writing generated Go source files into ./datasets/.
//
// Usage from repo root:
//
//	go run ./tools/fetch_datasets ausbeer    # populate ausbeer_r.go
//	go run ./tools/fetch_datasets gasoline   # populate gasoline_r.go
//	go run ./tools/fetch_datasets all        # both
//
// Generated files (ausbeer_r.go, gasoline_r.go) are MIT-licensed alongside
// the rest of goarima; the underlying numerical values are factual data
// from public-government publishers (ABS, US EIA), not subject to copyright
// per Feist v Rural Telephone (1991).
//
// The fetch URLs in this file point at well-known public mirrors of the
// data — when those break, update the URL constants and re-run. The tool
// validates fetched data's shape (obs count, value range) before writing,
// so a broken upstream produces a clear error rather than silently bad
// data.
package main

import (
	"errors"
	"fmt"
	"io"
	"net/http"
	"os"
	"strconv"
	"strings"
	"time"
)

// Canonical sources via the Rdatasets project (Vincent Arel-Bundock's
// CSV mirror of standard R datasets — apache-2.0). The fpp2 package
// versions match Hyndman's textbook and R::forecast as of fpp2's
// release. If these URLs break, alternates: fma/ausbeer.csv, or
// download from the original ABS / US EIA publishers and parse.
const (
	ausbeerURL  = "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/ausbeer.csv"
	gasolineURL = "https://vincentarelbundock.github.io/Rdatasets/csv/fpp2/gasoline.csv"
)

// Expected shape sanity-checks. If fetched data falls outside these
// bounds we abort rather than write nonsense.
const (
	ausbeerMinObs = 200
	ausbeerMaxObs = 240
	ausbeerMinVal = 100.0 // megalitres; smallest historical quarter
	ausbeerMaxVal = 700.0

	gasolineMinObs = 1000
	gasolineMaxObs = 2000
	// fpp2 publishes gasoline in millions of barrels/day. pmdarima ships
	// the same series in thousands. We scale to thousands at fetch time
	// (see fetchGasoline) to match goarima's existing LoadGasoline unit
	// convention; bounds below check the post-scale values.
	gasolineScale  = 1000.0
	gasolineMinVal = 5000.0
	gasolineMaxVal = 12000.0
)

func main() {
	if len(os.Args) < 2 {
		fmt.Fprintln(os.Stderr, "usage: fetch_datasets {ausbeer|gasoline|all}")
		os.Exit(2)
	}
	target := os.Args[1]
	hasErr := false
	if target == "ausbeer" || target == "all" {
		if err := fetchAusbeer(); err != nil {
			fmt.Fprintf(os.Stderr, "ausbeer: %v\n", err)
			hasErr = true
		}
	}
	if target == "gasoline" || target == "all" {
		if err := fetchGasoline(); err != nil {
			fmt.Fprintf(os.Stderr, "gasoline: %v\n", err)
			hasErr = true
		}
	}
	if hasErr {
		os.Exit(1)
	}
}

func fetchAusbeer() error {
	body, err := httpGet(ausbeerURL)
	if err != nil {
		return fmt.Errorf("fetch %s: %w", ausbeerURL, err)
	}
	values, err := parseRdatasetsCSV(body)
	if err != nil {
		return fmt.Errorf("parse ausbeer csv: %w", err)
	}
	if err := validateShape(values, ausbeerMinObs, ausbeerMaxObs, ausbeerMinVal, ausbeerMaxVal); err != nil {
		return fmt.Errorf("ausbeer shape: %w", err)
	}
	return writeGenerated(
		"datasets/ausbeer_r.go",
		"loadAusbeerR",
		"Australian quarterly beer production matching R's forecast::ausbeer.",
		"Source: Australian Bureau of Statistics; mirrored via fpp2 textbook.",
		ausbeerURL,
		values,
	)
}

func fetchGasoline() error {
	body, err := httpGet(gasolineURL)
	if err != nil {
		return fmt.Errorf("fetch %s: %w", gasolineURL, err)
	}
	values, err := parseRdatasetsCSV(body)
	if err != nil {
		return fmt.Errorf("parse gasoline csv: %w", err)
	}
	// fpp2 → thousands-of-barrels (matches goarima's LoadGasoline convention).
	for i := range values {
		values[i] *= gasolineScale
	}
	if err := validateShape(values, gasolineMinObs, gasolineMaxObs, gasolineMinVal, gasolineMaxVal); err != nil {
		return fmt.Errorf("gasoline shape: %w", err)
	}
	return writeGenerated(
		"datasets/gasoline_r.go",
		"loadGasolineForecastR",
		"Weekly US finished motor gasoline product supplied matching R's forecast::gasoline (thousands of barrels/day).",
		"Source: US Energy Information Administration; mirrored via Rdatasets/fpp2. Scaled ×1000 to match goarima's LoadGasoline unit convention.",
		gasolineURL,
		values,
	)
}

func httpGet(url string) ([]byte, error) {
	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Get(url)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("HTTP %d", resp.StatusCode)
	}
	return io.ReadAll(resp.Body)
}

// parseRdatasetsCSV parses the Rdatasets-mirror CSV format:
//
//	rownames,time,value
//	1,1956,284
//	2,1956.25,213
//	...
//
// Returns the `value` column, skipping the header row. Tolerates
// blank/NA values (treated as series termination, since R's ts can have
// NAs but our []float64 doesn't).
func parseRdatasetsCSV(body []byte) ([]float64, error) {
	lines := strings.Split(strings.ReplaceAll(string(body), "\r\n", "\n"), "\n")
	if len(lines) == 0 {
		return nil, errors.New("empty CSV")
	}
	// Header check: expect "rownames,time,value" or similar.
	header := strings.Split(strings.TrimSpace(lines[0]), ",")
	if len(header) < 3 {
		return nil, fmt.Errorf("expected ≥3 cols (rownames,time,value); got header %q", lines[0])
	}
	out := make([]float64, 0, len(lines)-1)
	for i, line := range lines[1:] {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ",")
		if len(fields) < 3 {
			return nil, fmt.Errorf("line %d: only %d columns: %q", i+1, len(fields), line)
		}
		valStr := strings.TrimSpace(fields[2])
		if valStr == "" || valStr == "NA" {
			continue
		}
		v, err := strconv.ParseFloat(valStr, 64)
		if err != nil {
			return nil, fmt.Errorf("line %d: parse %q: %w", i+1, valStr, err)
		}
		out = append(out, v)
	}
	if len(out) == 0 {
		return nil, errors.New("no data rows parsed")
	}
	return out, nil
}

func validateShape(values []float64, minObs, maxObs int, minVal, maxVal float64) error {
	if len(values) < minObs || len(values) > maxObs {
		return fmt.Errorf("got %d obs, expected [%d, %d]", len(values), minObs, maxObs)
	}
	for i, v := range values {
		if v < minVal || v > maxVal {
			return fmt.Errorf("idx %d: value %g outside expected [%g, %g]",
				i, v, minVal, maxVal)
		}
	}
	return nil
}

func writeGenerated(path, varName, descr, attrib, url string, values []float64) error {
	var sb strings.Builder
	fmt.Fprintf(&sb, "// Code generated by tools/fetch_datasets. DO NOT EDIT.\n")
	fmt.Fprintf(&sb, "//\n")
	fmt.Fprintf(&sb, "// %s\n", descr)
	fmt.Fprintf(&sb, "// %s\n", attrib)
	fmt.Fprintf(&sb, "// Fetched from: %s\n", url)
	fmt.Fprintf(&sb, "// Generated:    %s\n", time.Now().UTC().Format(time.RFC3339))
	fmt.Fprintf(&sb, "// Observations: %d\n\n", len(values))
	fmt.Fprintf(&sb, "package datasets\n\n")
	fmt.Fprintf(&sb, "func init() { %s = func() []float64 { return rData_%s } }\n\n", varName, varName)
	fmt.Fprintf(&sb, "var rData_%s = []float64{\n", varName)
	for i, v := range values {
		if i%8 == 0 {
			sb.WriteString("\t")
		}
		fmt.Fprintf(&sb, "%g,", v)
		if i%8 == 7 || i == len(values)-1 {
			sb.WriteString("\n")
		} else {
			sb.WriteString(" ")
		}
	}
	sb.WriteString("}\n")

	if err := os.WriteFile(path, []byte(sb.String()), 0o644); err != nil {
		return fmt.Errorf("write %s: %w", path, err)
	}
	fmt.Printf("✓ wrote %s (%d obs)\n", path, len(values))
	return nil
}
