package datasets

// LoadWWWusage returns Internet usage per minute (n=100), R built-in WWWusage.
// Time series of the numbers of users connected to the Internet through a server,
// every minute (Makridakis et al. 1998). Used in `forecast`'s test-newarima2.R.
func LoadWWWusage() []float64 {
	return []float64{
		88, 84, 85, 85, 84, 85, 83, 85, 88, 89,
		91, 99, 104, 112, 126, 138, 146, 151, 150, 148,
		147, 149, 143, 132, 131, 139, 147, 150, 148, 145,
		140, 134, 131, 131, 129, 126, 126, 132, 137, 140,
		142, 150, 159, 167, 170, 171, 172, 172, 174, 175,
		172, 172, 174, 174, 169, 165, 156, 142, 131, 121,
		112, 104, 102, 99, 99, 95, 88, 84, 84, 87,
		89, 88, 85, 86, 89, 91, 91, 94, 101, 110,
		121, 135, 145, 149, 156, 165, 171, 175, 177, 182,
		193, 204, 208, 210, 215, 222, 228, 226, 222, 220,
	}
}
