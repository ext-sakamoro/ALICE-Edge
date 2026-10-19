// expect-ir: flag
// expect-syn: flag
// targets: any
// the other side of probes 48 and 49: the negated result is also used on its own, so the
// negation survives to the IR as an fneg of the fma and the IR half refuses it
const INV_60: f32 = 1.0 / 60.0;
pub fn probe(deg: f32, raw: f32) -> (f32, f32) {
    let min = -(deg.mul_add(100.0, -raw));
    (deg + min * INV_60, min)
}
