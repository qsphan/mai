/-
`sys_wait` meets its specification, given `argaddr` and `kwait`.
-/
import Xv6.ProofSysWait

namespace Xv6

theorem SysWait (AA : ARGADDR) (KW : KWAIT) : SYSWAIT := sys_wait_proof AA KW

end Xv6
