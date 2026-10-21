---
id: KB-0002
title: Resetting Your Halcyon ID Password
category: how-to
service: Identity
tags: [password, halcyon-id, account, lockout, self-service]
owner: Identity & Access Management
last_updated: 2026-07-02
---

# Resetting Your Halcyon ID Password

Your Halcyon ID is the single sign-on account used for email, Teams, the HR Portal, the VPN
and most internal applications. Changing it in one place changes it everywhere.

## Password requirements

- Minimum length: **14 characters**.
- Must not contain your name, username, or the word "Halcyon".
- Must not match any of your last **12** passwords.
- Passphrases are encouraged, for example `correct-river-lantern-42`.
- Passwords do **not** expire on a schedule. You only need to change your password if it is
  forgotten, suspected to be compromised, or you are prompted by the Security team.

## Self-service reset (recommended)

1. Go to `https://id.halcyon-dynamics.example/reset` from any browser.
2. Enter your work email address.
3. Verify your identity with your registered MFA method (Authenticator app or FIDO2 key).
4. Choose a new password that meets the requirements above.
5. Sign out and back in on all devices, including your phone's email app, to avoid lockouts.

Self-service reset requires that you have already enrolled in MFA. If you have never set up
MFA, you must contact the Service Desk (see below).

## Account lockout

Your account is locked for **15 minutes** after **5 failed sign-in attempts**. The lock
clears automatically; you do not need to raise a ticket. A common cause of repeated lockouts
is an old password saved on a phone or a mapped drive. Update or remove saved credentials
after every password change.

## Assisted reset by the Service Desk

If you cannot use self-service:

1. Call the Service Desk on extension 4357 or visit the walk-up IT desk.
2. The analyst will verify your identity using your employee ID and a callback to the phone
   number in the HR system. **Analysts will never ask for your current password.**
3. You will receive a temporary password that must be changed at first sign-in.

## Suspected compromise

If you think someone else knows your password or you entered it on a suspicious website,
reset it immediately using self-service and then report it as a security incident through
the portal under **Security > Report a Phishing or Compromise**. Compromised accounts are
treated as a P2 incident at minimum.
