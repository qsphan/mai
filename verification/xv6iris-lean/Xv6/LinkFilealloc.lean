/-
`filealloc` meets its interface, given the interfaces of `acquire` and
`release`.  This file imports `ProofFilealloc` (the only Proof file it may
import).
-/
import Xv6.ProofFilealloc

namespace Xv6

theorem Filealloc (AC : ACQUIRE) (RE : RELEASE) : FILEALLOC := filealloc_proof AC RE

end Xv6
