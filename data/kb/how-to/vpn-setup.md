---
id: KB-0001
title: Connecting to the Corporate VPN (SecureLink)
category: how-to
service: VPN
tags: [vpn, securelink, remote-access, network]
owner: Network Operations
last_updated: 2026-08-14
---

# Connecting to the Corporate VPN (SecureLink)

SecureLink is the only approved VPN client for accessing Halcyon Dynamics internal systems
from outside the office. Internal resources such as the HR Portal, shared drives and the
finance system are not reachable from the public internet without an active VPN session.

## Before you start

- You need an active Halcyon ID account with MFA enrolled (see KB-0003).
- Company-managed laptops have SecureLink pre-installed. Personal devices are **not**
  permitted to connect to the VPN (see the Acceptable Use Policy, POL-0001).
- You need a working internet connection. Hotel and airport Wi-Fi captive portals must be
  accepted in a browser before SecureLink can connect.

## Connecting on Windows

1. Click the SecureLink shield icon in the system tray. If it is missing, open
   **Start > SecureLink**.
2. Confirm the gateway is set to `vpn.halcyon-dynamics.example`.
3. Click **Connect** and sign in with your Halcyon ID email and password.
4. Approve the push notification in Microsoft Authenticator, or tap your FIDO2 security key.
5. The shield icon turns green when connected. Connection status shows **Connected – Full Tunnel**.

## Connecting on macOS

1. Open SecureLink from the menu bar or from **Applications > SecureLink**.
2. Confirm the gateway is `vpn.halcyon-dynamics.example`.
3. Click **Connect**, sign in, and approve the MFA prompt.
4. If macOS asks to allow a VPN configuration, click **Allow** and enter your Mac password.

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| "Gateway unreachable" | Captive portal or firewall blocking UDP 4501 | Open a browser to accept the Wi-Fi portal; try the fallback gateway `vpn-tcp.halcyon-dynamics.example` |
| MFA prompt never arrives | Phone offline or notifications disabled | Use the 6-digit code in Authenticator instead of the push |
| Connects then drops every few minutes (macOS) | Known issue KI-0007 | See KB article KI-0007 for the workaround |
| "Certificate not trusted" | Device not enrolled in device management | Contact the Service Desk; the laptop must be re-enrolled |

If the VPN is down for many users at once, check the service status page at
`https://status.halcyon-dynamics.example` before raising a ticket. A widespread VPN outage
is handled as a P1 incident by Network Operations.

## Getting help

Raise a ticket in the Service Desk portal (`https://help.halcyon-dynamics.example`) under
**Network > VPN**, or call the Service Desk on extension 4357 (HELP).
