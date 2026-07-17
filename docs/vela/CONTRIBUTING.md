<!--
SPDX-FileCopyrightText: Copyright 2020-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

SPDX-License-Identifier: Apache-2.0

Licensed under the Apache License, Version 2.0 (the License); you may
not use this file except in compliance with the License.
You may obtain a copy of the License at

www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an AS IS BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
-->
# Vela Contributions

Contributions to Vela are very much welcomed!

## Coding Standard

All code must run through the developer pre-commit checks described in [Testing](TESTING.md)

## Submitting

### One-time Registration
In order to submit a contribution to the project you must first complete the
following one-time process:
* Sign in
   * Requires a GitHub ID
* Add SSH key
   * This is added under the user settings page
   * If there is a problem then make sure there is a valid email address in the
   Email Addresses field
* Request permission to create project forks
   * See [Requesting Access](https://gitlab.arm.com/documentation/contributions#requesting-access)
* Create a fork of the [Vela project](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela)
   * Click the fork button in the top-right corner on the project's main page

### Process

Contributions are made by creating a merge request from your fork to the branch
 `main` of the target project `artificial-intelligence/ethos-u/ethos-u-vela`.
 Every commit in the merge request must include a Signed-off-by in the commit
 message to indicate that you agree to the guidelines below.

## Guidelines

Contributions are only accepted under the following conditions:

* All code passes the developer pre-commit checks
* All code passes a review process conducted on the platform
* You certify that the origin of the submission conforms to the
[Developer Certificate of Origin (DCO) V1.1](https://developercertificate.org/)
* You give permission according to the [Apache License 2.0](LICENSE.txt)

To indicate that you agree to the above you need to add a Signed-off-by to your
commit using your real name and e-mail address.
e.g. 'Signed-off-by: Real Name \<username@example.org\>' to every commit
message.  This can be done automatically by adding the `-s` option to your
`git commit` command.

Contributions are not accepted from pseudonyms or anonymous sources.

## Code Reviews

All contributions go through a code review process.  Only submissions that are
approved and verified by this process will be accepted.  Code reviews are
performed using
[Arm's GitLab Instance](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela).

## Testing Prior to Submission

Prior to submitting a patch for review please make sure that all the pre-commit
checks and tests have been run and are passing (see [Vela Testing](TESTING.md)
for more details).

## Bug Resolution

In the case that your submission aims to resolve a bug, please follow the
[Bug Reporting Process](BUGS.md) and include a link to your patch in the issue
description.