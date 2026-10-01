/-
`argraw` meets its specification, given `myproc`.
-/
import Xv6.ProofArgraw

namespace Xv6

theorem Argraw (MP : MYPROC) : ARGRAW := argraw_proof MP

end Xv6
