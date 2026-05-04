// Diagnostic tool: profiles mutex contention during a parallel stepwise
// AutoArima run, prints the top-10 mutex hotspots so we can verify
// whether `cacheMu` (or any other goarima mutex) is a bottleneck.
//
// Usage:
//
//	go run ./cmd/lock_profile
//
// Prints lines like:
//
//	  total contention: 12.3ms
//	  cacheMu (auto.go:425):  3.4ms (28%)
//	  …
//
// Exits non-zero if the top hotspot exceeds a threshold of total wall
// time, signalling that the audit's "no measured contention" claim
// would need revisiting.
package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"
	"strings"
	"time"

	"github.com/hailam/goarima/arima"
	"github.com/hailam/goarima/datasets"
)

func main() {
	runtime.SetMutexProfileFraction(1) // sample every contention event

	ap := datasets.LoadAirPassengers()

	// Run several stepwise AutoArima calls to produce enough samples for
	// the profiler. Use parallel default (NJobs=0) so the cache mutex is
	// actually contended.
	const reps = 20
	start := time.Now()
	for i := 0; i < reps; i++ {
		_, err := arima.AutoArima(ap, nil, arima.AutoArimaOpts{
			M: 12, MaxP: 3, MaxQ: 3, MaxCapP: 1, MaxCapQ: 1,
			MaxOrder: 5, MaxD: 2, IC: arima.AICc, MaxIter: 50,
		})
		if err != nil {
			fmt.Fprintln(os.Stderr, "fit error:", err)
			os.Exit(1)
		}
	}
	wall := time.Since(start)
	fmt.Printf("ran %d stepwise AutoArima calls in %v (%.2f ms each)\n",
		reps, wall, float64(wall.Nanoseconds())/float64(reps)/1e6)

	// Dump the mutex profile and look for goarima-internal hotspots.
	mp := pprof.Lookup("mutex")
	if mp == nil {
		fmt.Fprintln(os.Stderr, "mutex profile not available")
		os.Exit(1)
	}
	var buf strings.Builder
	if err := mp.WriteTo(&buf, 1); err != nil { // 1 = legacy text format
		fmt.Fprintln(os.Stderr, "profile write:", err)
		os.Exit(1)
	}
	out := buf.String()
	if out == "" {
		fmt.Println("no mutex contention recorded (everything below the sampling threshold)")
		return
	}

	// Print the raw legacy-format output. Look for `cacheMu` / `auto.go`.
	fmt.Println("---- mutex contention profile (legacy text format) ----")
	fmt.Println(out)

	// Headline: did goarima-internal mutexes appear?
	relevant := []string{"cacheMu", "auto.go", "github.com/hailam/goarima"}
	for _, kw := range relevant {
		count := strings.Count(out, kw)
		if count > 0 {
			fmt.Printf("found %q in profile: %d occurrence(s)\n", kw, count)
		}
	}
}
