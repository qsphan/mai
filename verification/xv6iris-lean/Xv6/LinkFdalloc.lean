/-
`fdalloc` meets its interface, given the interface of `myproc`.  This file
imports `ProofFdalloc` (the only Proof file it may import).
-/
import Xv6.ProofFdalloc

namespace Xv6

theorem Fdalloc (MP : MYPROC) : FDALLOC := fdalloc_proof MP

end Xv6
