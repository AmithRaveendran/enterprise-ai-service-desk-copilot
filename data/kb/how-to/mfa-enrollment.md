---
id: KB-0003
title: Enrolling in Multi-Factor Authentication (MFA)
category: how-to
service: Identity
tags: [mfa, authenticator, fido2, security-key, halcyon-id]
owner: Identity & Access Management
last_updated: 2026-06-20
---

# Enrolling in Multi-Factor Authentication (MFA)

MFA is mandatory for every Halcyon ID account. New starters must enrol within **7 days** of
their start date; after that, sign-in to all applications is blocked until enrolment is
complete.

## Approved methods

| Method | Approved | Notes |
|---|---|---|
| Microsoft Authenticator (push + number matching) | Yes | Default for most staff |
| FIDO2 hardware security key | Yes | Required for administrators and Finance staff |
| Authenticator 6-digit code (TOTP) | Yes | Fallback when push is unavailable |
| SMS text message | **No** | Not permitted; vulnerable to SIM-swap attacks |
| Voice call | **No** | Not permitted |

## Enrolling with Microsoft Authenticator

1. Install **Microsoft Authenticator** from the App Store or Google Play on your phone.
2. On your computer, go to `https://id.halcyon-dynamics.example/security`.
3. Select **Add sign-in method > Authenticator app** and click **Next**.
4. In the app, tap **+ > Work or school account > Scan QR code** and scan the code on screen.
5. Approve the test notification and type the number shown on your computer.

## Enrolling a FIDO2 security key

Security keys are issued by the Service Desk. Request one in the portal under
**Hardware > Security Key**.

1. Go to `https://id.halcyon-dynamics.example/security`.
2. Select **Add sign-in method > Security key** and choose USB or NFC.
3. Insert or tap the key, create a PIN when prompted, and touch the key's sensor.
4. Name the key (for example "YubiKey – blue") so you can identify it later.

We recommend registering **two** methods so you are not locked out if you lose a device.

## Lost or replaced phone

If you get a new phone, add it as a method **before** wiping the old one. If your phone is
lost or stolen, call the Service Desk on extension 4357. After identity verification, the
analyst can issue a **Temporary Access Pass** valid for 8 hours so you can register a new
method.
