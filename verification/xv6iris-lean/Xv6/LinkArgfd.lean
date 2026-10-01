/-
`argfd` meets its specification, given `argint` and `myproc`.
-/
import Xv6.ProofArgfd

namespace Xv6

theorem Argfd (AI : ARGINT) (MP : MYPROC) : ARGFD := argfd_proof AI MP

end Xv6
