/-
`argaddr` meets its specification, given `myproc` and `argraw`.
-/
import Xv6.ProofArgaddr

namespace Xv6

theorem Argaddr (MP : MYPROC) (AR : ARGRAW) : ARGADDR := argaddr_proof MP AR

end Xv6
