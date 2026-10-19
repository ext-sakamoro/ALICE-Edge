// expect-ir: clean
// expect-syn: flag
// targets: any
// the negation's only use is a multiply by a constant: LLVM folds it into the constant
// (`fmul (fma), -C`), so no fneg reaches the IR, whose text is then the conforming `fma * -C`
// (probe 41); the syntax-tree gate refuses this source form (G3 is defined on the source)
const INV_60: f32 = 1.0 / 60.0;
pub fn probe(a: f32, b: f32, c: f32) -> f32 {
    -(a.mul_add(b, c)) * INV_60
}
