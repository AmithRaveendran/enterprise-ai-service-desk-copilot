---
id: KB-0007
title: Connecting to Office Wi-Fi Networks
category: how-to
service: Wi-Fi
tags: [wifi, wireless, network, guest, office]
owner: Network Operations
last_updated: 2026-03-19
---

# Connecting to Office Wi-Fi Networks

Each Halcyon Dynamics office broadcasts three wireless networks.

| Network | Who can use it | Access |
|---|---|---|
| **Halcyon-Corp** | Company-managed laptops only | Automatic, using a device certificate. Full internal access |
| **Halcyon-BYOD** | Personal phones and tablets of employees | Sign in with Halcyon ID. Internet and Microsoft 365 only |
| **Halcyon-Guest** | Visitors | Daily voucher from reception. Internet only, 4-hour sessions |

## Company laptops (Halcyon-Corp)

Managed laptops connect to **Halcyon-Corp** automatically. You never need to enter a
password. If your laptop will not connect:

1. Turn Wi-Fi off and on again.
2. Restart the laptop while in the office so the device certificate can renew.
3. If the laptop has been away from the office and off the VPN for more than **90 days**,
   the certificate may have expired. Connect with a network cable or the VPN once, then
   restart.

## Personal devices (Halcyon-BYOD)

1. Select **Halcyon-BYOD**.
2. Sign in with your Halcyon ID email and password, and approve the MFA prompt.
3. Accept the acceptable use terms.

Personal devices cannot reach internal systems such as shared drives or the HR Portal over
Halcyon-BYOD. Up to **3** personal devices may be registered per person.

## Guests (Halcyon-Guest)

Reception issues guest vouchers valid for one day. Employees must not share their own
credentials with visitors. Guest traffic is filtered and isolated from the corporate
network.

## Slow or unstable Wi-Fi

- Move closer to an access point; meeting rooms with closed doors often have weaker signal.
- Check `https://status.halcyon-dynamics.example` for a Wi-Fi incident at your site.
- Report persistent problems under **Network > Wi-Fi Issue** and include your office,
  floor and the time the problem happened.
