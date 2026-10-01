/-
`sys_pause` meets its specification, given `argint`, `acquire`, `release`,
`myproc`, `killed`, `sleep_prepare` and `sleep`.
-/
import Xv6.ProofSysPause

namespace Xv6

theorem SysPause (AI : ARGINT) (AC : ACQUIRE) (RE : RELEASE) (MP : MYPROC) (KL : KILLED)
    (SP : SLEEP_PREPARE) (SL : SLEEP) : SYSPAUSE := sys_pause_proof AI AC RE MP KL SP SL

end Xv6
