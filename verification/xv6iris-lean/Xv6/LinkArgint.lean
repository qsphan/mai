/-
`argint` meets its specification, given `myproc` and `argraw`.
-/
import Xv6.ProofArgint

namespace Xv6

theorem Argint (MP : MYPROC) (AR : ARGRAW) : ARGINT := argint_proof MP AR

end Xv6
