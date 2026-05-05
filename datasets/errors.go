package datasets

import "fmt"

// errDatasetNotFetched returns a clear error when an on-demand R-variant
// dataset hasn't been populated yet. The message points users at the
// fetch tool with the exact command to run.
func errDatasetNotFetched(name, fetchArg string) error {
	return fmt.Errorf(
		"dataset %q not fetched yet — run `go run ./tools/fetch_datasets %s` from the repo root to populate",
		name, fetchArg,
	)
}
