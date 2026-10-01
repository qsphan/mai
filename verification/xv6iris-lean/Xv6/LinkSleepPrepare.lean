/-
Link `sleep_prepare`: the proof instance clients import.  `sleep_prepare`
calls `myproc`, `acquire` and `release`; the interfaces stay parameters
here, so a client may close them with the linked ones (`LinkMyproc`,
`LinkAcquire`, `LinkRelease`) or with its own.
-/
import Xv6.ProofSleepPrepare

namespace Xv6

/-- The proved `sleep_prepare` interface, given `myproc`, `acquire` and
`release`. -/
theorem SleepPrepare (MP : MYPROC) (AC : ACQUIRE) (RE : RELEASE) : SLEEP_PREPARE :=
  sleep_prepare_proof MP AC RE

end Xv6
