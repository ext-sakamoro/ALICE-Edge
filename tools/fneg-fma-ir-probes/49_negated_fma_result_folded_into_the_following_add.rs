// expect-ir: clean
// expect-syn: flag
// targets: any
// the shape found in ALICE-Edge sensors.rs (latitude minutes): the negated result is multiplied
// by a constant and added, and LLVM folds the negation into the add (`fsub deg, (fma * C)`), so no
// fneg reaches the IR; the syntax-tree gate refuses this source form
const INV_60: f32 = 1.0 / 60.0;
pub fn probe(deg: f32, raw: f32) -> f32 {
    let min = -(deg.mul_add(100.0, -raw));
    deg + min * INV_60
}
