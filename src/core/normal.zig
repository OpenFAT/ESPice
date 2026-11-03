//! The standard normal distribution: its quantile Φ⁻¹, its distribution
//! function Φ and its upper tail Q through the complementary error function.
//! Scalars in, a scalar out. Re-exported by `numerics.zig`.
//! Cites: musl src/math/erf.c (`erfc`, from FreeBSD s_erf.c); P. J. Acklam's
//! rational approximation of the normal quantile.
//!
//! Contents:
//! - `pub fn normalQuantile(u: f64) f64`
//! - `pub fn normalCdf(x: f64) f64`
//! - `pub fn normalTail(x: f64) f64`
//! - `pub fn erfc(x: f64) f64`

const std = @import("std");

// musl/FreeBSD names: erx, pp0..pp4, qq1..qq5 (each denominator here leads
// with its constant 1), pa0..pa6, qa1..qa6, ra0..ra7, sa1..sa8, rb0..rb6,
// sb1..sb7.
const erf_erx = 8.45062911510467529297e-01;
const erf_pp = [_]f64{ 1.28379167095512558561e-01, -3.25042107247001499370e-01, -2.84817495755985104766e-02, -5.77027029648944159157e-03, -2.37630166566501626084e-05 };
const erf_qq = [_]f64{ 1, 3.97917223959155352819e-01, 6.50222499887672944485e-02, 5.08130628187576562776e-03, 1.32494738004321644526e-04, -3.96022827877536812320e-06 };
const erf_pa = [_]f64{ -2.36211856075265944077e-03, 4.14856118683748331666e-01, -3.72207876035701323847e-01, 3.18346619901161753674e-01, -1.10894694282396677476e-01, 3.54783043256182359371e-02, -2.16637559486879084300e-03 };
const erf_qa = [_]f64{ 1, 1.06420880400844228286e-01, 5.40397917702171048937e-01, 7.18286544141962662868e-02, 1.26171219808761642112e-01, 1.36370839120290507362e-02, 1.19844998467991074170e-02 };
const erf_ra = [_]f64{ -9.86494403484714822705e-03, -6.93858572707181764372e-01, -1.05586262253232909814e+01, -6.23753324503260060396e+01, -1.62396669462573470355e+02, -1.84605092906711035994e+02, -8.12874355063065934246e+01, -9.81432934416914548592e+00 };
const erf_sa = [_]f64{ 1, 1.96512716674392571292e+01, 1.37657754143519042600e+02, 4.34565877475229228821e+02, 6.45387271733267880336e+02, 4.29008140027567833386e+02, 1.08635005541779435134e+02, 6.57024977031928170135e+00, -6.04244152148580987438e-02 };
const erf_rb = [_]f64{ -9.86494292470009928597e-03, -7.99283237680523006574e-01, -1.77579549177547519889e+01, -1.60636384855821916062e+02, -6.37566443368389627722e+02, -1.02509513161107724954e+03, -4.83519191608651397019e+02 };
const erf_sb = [_]f64{ 1, 3.03380607434824582924e+01, 3.25792512996573918826e+02, 1.53672958608443695994e+03, 3.19985821950859553908e+03, 2.55305040643316442583e+03, 4.74528541206955367215e+02, -2.24409524465858183362e+01 };

/// The standard normal quantile Φ⁻¹(u), u in (0, 1): Acklam's rational
/// approximation, relative error below 1.2e-9. At 0 or 1 it returns NaN,
/// so callers draw from the open interval.
pub fn normalQuantile(u: f64) f64 {
    const a = [_]f64{ -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02, 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00 };
    const b = [_]f64{ -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02, 6.680131188771972e+01, -1.328068155288572e+01 };
    const c = [_]f64{ -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00, -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00 };
    const d = [_]f64{ 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00, 3.754408661907416e+00 };
    const low = 0.02425;
    if (u < low or u > 1 - low) {
        const q = @sqrt(-2 * @log(if (u < low) u else 1 - u));
        const x = (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) /
            ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1);
        return if (u < low) x else -x;
    }
    const q = u - 0.5;
    const r = q * q;
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q /
        (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1);
}

/// Φ(x), the standard normal distribution function.
pub fn normalCdf(x: f64) f64 {
    return 0.5 * erfc(-x / std.math.sqrt2);
}

/// Q(x) = 1 - Φ(x), the upper tail, with its relative precision kept far
/// out (Q(6) is 1e-9), where 1 - Φ(x) cancels.
pub fn normalTail(x: f64) f64 {
    return 0.5 * erfc(x / std.math.sqrt2);
}

/// The complementary error function, below 1 ulp: a port of musl
/// src/math/erf.c `erfc` (from FreeBSD s_erf.c), coefficients as there.
pub fn erfc(x: f64) f64 {
    const bits: u64 = @bitCast(x);
    const ix: u32 = @as(u32, @truncate(bits >> 32)) & 0x7fff_ffff;
    const negative = bits >> 63 != 0;
    // erfc(nan) = nan, erfc(+inf) = 0, erfc(-inf) = 2.
    if (ix >= 0x7ff0_0000) return @as(f64, if (negative) 2 else 0) + 1 / x;
    if (ix < 0x3feb_0000) { // |x| < 0.84375
        if (ix < 0x3c70_0000) return 1 - x; // |x| < 2^-56
        const z = x * x;
        const y = horner(&erf_pp, z) / horner(&erf_qq, z);
        if (negative or ix < 0x3fd0_0000) return 1 - (x + x * y); // x < 1/4
        return 0.5 - (x - 0.5 + x * y);
    }
    if (ix < 0x403c_0000) return if (negative) 2 - erfcAbove(ix, x) else erfcAbove(ix, x); // |x| < 28
    return if (negative) 2 else 0;
}

// ---- private ----

/// erfc(|x|) for 0.84375 <= |x| < 28; `ix` is |x|'s high word.
fn erfcAbove(ix: u32, x: f64) f64 {
    const ax = @abs(x);
    if (ix < 0x3ff4_0000) { // |x| < 1.25
        const s = ax - 1;
        return 1 - erf_erx - horner(&erf_pa, s) / horner(&erf_qa, s);
    }
    const s = 1 / (ax * ax);
    const rs: [2]f64 = if (ix < 0x4006_db6d) // |x| < 1/0.35
        .{ horner(&erf_ra, s), horner(&erf_sa, s) }
    else
        .{ horner(&erf_rb, s), horner(&erf_sb, s) };
    // |x| with its low 32 bits cleared, so z*z is exact.
    const z: f64 = @bitCast(@as(u64, @bitCast(ax)) & 0xffff_ffff_0000_0000);
    return @exp(-z * z - 0.5625) * @exp((z - ax) * (z + ax) + rs[0] / rs[1]) / ax;
}

/// c[0] + x*(c[1] + x*(... + x*c[n-1])), in musl's order of operations.
fn horner(c: []const f64, x: f64) f64 {
    var acc = c[c.len - 1];
    var i = c.len - 1;
    while (i > 0) {
        i -= 1;
        acc = c[i] + x * acc;
    }
    return acc;
}

test "normal tail: mpmath values in every erfc branch, and erfc at the edges" {
    // Q(x) = mpmath.ncdf(-x) at 30 digits, rounded to 16. The x cover the
    // branches of erfc(x/sqrt2): below 1/4, below 0.84375, below 1.25,
    // below 1/0.35 and up to 28.
    for ([_][2]f64{
        .{ 0.2, 0.42074029056089696 },  .{ 0.5, 0.30853753872598688 },  .{ 1, 0.15865525393145707 },
        .{ 1.5, 0.066807201268858057 }, .{ 2, 0.022750131948179207 },   .{ 3, 0.0013498980316300946 },
        .{ 5, 2.8665157187919391e-07 }, .{ 6, 9.8658764503769814e-10 }, .{ 8, 6.2209605742717841e-16 },
    }) |c| {
        try std.testing.expectApproxEqRel(c[1], normalTail(c[0]), 1e-13);
        try std.testing.expectApproxEqRel(c[1], normalCdf(-c[0]), 1e-13);
        try std.testing.expectApproxEqRel(1 - c[1], normalCdf(c[0]), 1e-15);
    }
    try std.testing.expectEqual(@as(f64, 0.5), normalTail(0));
    try std.testing.expectEqual(@as(f64, 0), erfc(std.math.inf(f64)));
    try std.testing.expectEqual(@as(f64, 2), erfc(-std.math.inf(f64)));
    try std.testing.expectEqual(@as(f64, 0), erfc(30));
    try std.testing.expect(std.math.isNan(erfc(std.math.nan(f64))));
}

test "normal tail and quantile invert each other: an independent approximation" {
    // Acklam's quantile shares nothing with erfc; its relative error is 1.2e-9.
    var x: f64 = 0;
    while (x <= 8) : (x += 0.125) try std.testing.expectApproxEqAbs(-x, normalQuantile(normalTail(x)), 1e-8 * @max(1, x));
}

test "normal quantile: symmetric, and the 97.5% point is 1.96" {
    try std.testing.expectApproxEqAbs(@as(f64, 0), normalQuantile(0.5), 1e-12);
    try std.testing.expectApproxEqAbs(@as(f64, 1.959963984540054), normalQuantile(0.975), 1e-8);
    try std.testing.expectApproxEqAbs(-normalQuantile(0.01), normalQuantile(0.99), 1e-12);
}

test "normal quantile is monotone across the tail seam" {
    var prev = normalQuantile(1e-12);
    for ([_]f64{ 1e-6, 0.02, 0.02425, 0.0243, 0.3, 0.5, 0.7, 0.97575, 0.98, 1 - 1e-12 }) |u| {
        const x = normalQuantile(u);
        try std.testing.expect(x > prev);
        prev = x;
    }
    try std.testing.expect(std.math.isNan(normalQuantile(0)));
}
