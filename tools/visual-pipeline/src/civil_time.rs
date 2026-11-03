//! ISO-8601 UTC text and Unix epoch milliseconds in the proleptic Gregorian calendar.

/// Format Unix epoch milliseconds as an ISO-8601 UTC string (`YYYY-MM-DDTHH:MM:SS+00:00` or `.sss`).
pub fn unix_ms_to_iso8601(unix_ms: i64) -> String {
    let total_secs = unix_ms.div_euclid(1000);
    let millis = unix_ms.rem_euclid(1000);
    let days = total_secs.div_euclid(86_400);
    let sec_of_day = total_secs.rem_euclid(86_400);
    let hour = sec_of_day / 3600;
    let minute = (sec_of_day % 3600) / 60;
    let second = sec_of_day % 60;
    let (year, month, day) = civil_from_days(days);
    if millis == 0 {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}+00:00")
    } else {
        format!("{year:04}-{month:02}-{day:02}T{hour:02}:{minute:02}:{second:02}.{millis:03}+00:00")
    }
}

/// Parse an ISO-8601 timestamp (`YYYY-MM-DDTHH:MM:SS[.fff][Z|+HH:MM|-HH:MM]`) into Unix milliseconds.
pub fn parse_iso8601_ms(text: &str) -> Option<i64> {
    let s = text.trim();
    if s.len() < 19 {
        return None;
    }
    let year: i64 = s.get(0..4)?.parse().ok()?;
    let month: u32 = s.get(5..7)?.parse().ok()?;
    let day: u32 = s.get(8..10)?.parse().ok()?;
    let hour: i64 = s.get(11..13)?.parse().ok()?;
    let minute: i64 = s.get(14..16)?.parse().ok()?;
    let second: i64 = s.get(17..19)?.parse().ok()?;
    let rest = &s[19..];

    let (frac_ms, tz_part) = if let Some(after_dot) = rest.strip_prefix('.') {
        let digits: String = after_dot
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let tz = &after_dot[digits.len()..];
        let mut padded = digits;
        while padded.len() < 3 {
            padded.push('0');
        }
        padded.truncate(3);
        let ms: i64 = padded.parse().unwrap_or(0);
        (ms, tz)
    } else {
        (0, rest)
    };

    let offset_secs: i64 = if tz_part.is_empty() || tz_part == "Z" || tz_part == "z" {
        0
    } else if (tz_part.starts_with('+') || tz_part.starts_with('-')) && tz_part.len() >= 5 {
        let sign = if tz_part.starts_with('-') { -1 } else { 1 };
        let oh: i64 = tz_part.get(1..3)?.parse().ok()?;
        let om_start = if tz_part.as_bytes().get(3) == Some(&b':') {
            4
        } else {
            3
        };
        let om: i64 = tz_part.get(om_start..om_start + 2)?.parse().ok()?;
        sign * (oh * 3600 + om * 60)
    } else {
        0
    };

    let days = days_from_civil(year, month, day)?;
    let utc_secs = days * 86_400 + hour * 3600 + minute * 60 + second - offset_secs;
    Some(utc_secs * 1000 + frac_ms)
}

fn civil_from_days(z_raw: i64) -> (i64, u32, u32) {
    let z = z_raw + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u64;
    let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146_096) / 365;
    let y = (yoe as i64) + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m as u32, d as u32)
}

fn days_from_civil(y_raw: i64, m: u32, d: u32) -> Option<i64> {
    if !(1..=12).contains(&m) || !(1..=31).contains(&d) {
        return None;
    }
    let y = y_raw - i64::from(m <= 2);
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let mp = if m > 2 {
        u64::from(m - 3)
    } else {
        u64::from(m + 9)
    };
    let doy = (153 * mp + 2) / 5 + u64::from(d) - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    Some(era * 146_097 + (doe as i64) - 719_468)
}
