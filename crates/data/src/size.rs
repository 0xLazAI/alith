use bytesize::ByteSize;

/// Convert size to human-readable representations.
#[inline]
pub fn format_size(size: usize) -> String {
    format!("{}", ByteSize::b(size as u64))
}

#[cfg(test)]
mod tests {
    use crate::size::format_size;

    #[test]
    fn test_format_size() {
        let txt = "Your Data";
        assert_eq!(format_size(txt.len()), "9 B");
    }
}
