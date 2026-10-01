/-
`filedup` meets its specification, given `acquire` and `release`.
-/
import Xv6.ProofFiledup

namespace Xv6

theorem Filedup (AC : ACQUIRE) (RE : RELEASE) : FILEDUP := filedup_proof AC RE

end Xv6
