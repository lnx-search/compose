use triple_accel::levenshtein::levenshtein_simd_k;

#[inline]
pub fn compare(string: &str, other: &str, max_distance: i64) -> Option<i64> {
    levenshtein_simd_k(
        string.as_ref(),
        other.as_ref(),
        max_distance as u32,
    ).map(|v| v as i64)
}
