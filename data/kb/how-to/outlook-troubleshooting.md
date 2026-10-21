---
id: KB-0005
title: Troubleshooting Outlook and Email Problems
category: how-to
service: Email
tags: [email, outlook, mailbox, calendar, sync]
owner: Collaboration Services
last_updated: 2026-09-03
---

# Troubleshooting Outlook and Email Problems

Halcyon Dynamics uses Microsoft 365 for email and calendars. Before troubleshooting, check
`https://status.halcyon-dynamics.example` to see whether there is a known Email outage.

## Outlook will not open or is stuck on "Loading Profile"

1. Close Outlook completely, including from the system tray.
2. Start Outlook in safe mode: press **Windows + R**, type `outlook.exe /safe`, press Enter.
3. If safe mode works, a faulty add-in is the cause. Go to **File > Options > Add-ins** and
   disable recently added COM add-ins.
4. If safe mode also fails, create a new mail profile: **Control Panel > Mail > Show
   Profiles > Add**, then set it as the default.

## Emails are not arriving or sending

- Check the **Outbox** for stuck messages. Messages larger than **35 MB** including
  attachments are rejected. Use a OneDrive share link for large files.
- Check whether Outlook shows **Working Offline** in the status bar. Toggle it on the
  **Send / Receive** tab.
- Check your **Junk Email** and **Quarantine** folders. Quarantined messages can be released
  at `https://security.microsoft.com/quarantine`.

## Mailbox full

Standard mailboxes have a **100 GB** quota. When the mailbox is 98% full you can no longer
send email. To free space:

1. Empty **Deleted Items** and **Junk Email**.
2. Search for large items: in the search box type `size:>10MB`.
3. Enable the online archive, which is available to every user and has an additional 100 GB.

## Calendar or shared mailbox not syncing

- Remove and re-add the shared calendar or mailbox.
- Make sure **Cached Exchange Mode** is enabled under **File > Account Settings**.
- Access to a shared mailbox must be granted by the mailbox owner. Request access in the portal
  under **Email > Shared Mailbox Access**.

## Phone email

Only the **Outlook mobile app** is supported on phones. Native mail apps (iOS Mail, Gmail)
are blocked by policy. If Outlook mobile stops syncing after a password change, sign out of
the app and sign back in.

If none of these steps fix the problem, raise a ticket under **Email > Outlook Issue** and
attach a screenshot of any error message.
