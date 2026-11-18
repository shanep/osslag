//! Python's built-in `sum()` over floats.
//!
//! Since 3.12, CPython adds floats with Neumaier compensation, so a plain left fold can differ from
//! Python in the last bit. This follows `builtin_sum_impl` in CPython's `bltinmodule.c`.

pub fn py_sum(values: impl IntoIterator<Item = f64>) -> f64 {
    let mut total = 0.0f64;
    let mut c = 0.0f64;
    for x in values {
        let t = total + x;
        if total.abs() >= x.abs() {
            c += (total - t) + x;
        } else {
            c += (x - t) + total;
        }
        total = t;
    }
    if c != 0.0 && c.is_finite() {
        total += c;
    }
    total
}

#[cfg(test)]
mod tests {
    use super::py_sum;

    #[test]
    fn compensates_like_python() {
        // Python 3.12+: sum([0.1] * 10) == 1.0, a plain fold gives 0.9999999999999999.
        assert_eq!(py_sum([0.1; 10]), 1.0);
        assert_eq!(py_sum([1e100, 1.0, -1e100]), 1.0);
        assert_eq!(py_sum([]), 0.0);
    }
}
