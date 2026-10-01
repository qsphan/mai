/-
`sys_dup` meets its specification, given `argfd`, `fdalloc` and `filedup`.
-/
import Xv6.ProofSysDup

namespace Xv6

theorem SysDup (AF : ARGFD) (FD : FDALLOC) (FU : FILEDUP) : SYSDUP := sys_dup_proof AF FD FU

end Xv6
