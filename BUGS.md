<!--
SPDX-FileCopyrightText: Copyright 2021, 2023-2025 Arm Limited and/or its affiliates <open-source-office@arm.com>

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
# Vela Community Bug Reporting

If the Vela community encounters a bug, then we highly encourage the community
to report it. When issuing a report, use the
[Vela issue tracker](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/issues)
that is part of the Vela project on Arm's GitLab Instance. This site enables any
Vela user to discuss and share any Vela related issue with the community.

The only prerequisite for this process is a GitHub account, which will be used
when creating an account on Arm's GitLab Instance.

If you also have a fix for the bug you have identified, then we encourage
you to follow the [contributions guide](CONTRIBUTING.md).

The bug reporting process in its entirety is detailed below.

## Vela issue tracker

On the Vela issue tracker, one can add active bugs which will then be
addressable by the Vela community.

The system is accessible directly via the link above.

Anyone can browse the Vela issue tracker, since it is a public forum.
However, in order to file your own ticket you need to be logged in
to an account on Arm's GitLab Instance. These can be created by clicking
"GitHub" at <https://gitlab.arm.com/users/sign_in> and then linking a GitHub ID.

### Proprietary or Sensitive Information

As mentioned previously, the Vela issue tracker system is a public forum.
As such, anyone can see information posted to it.
Therefore, any proprietary information you wish to share should be done using
Arm's ML Model Review described below.

## Bug Reporting Process

The entire process of handling a bug is summarized in the following steps.

1. Discovering the bug
    * A user notices a discrepancy in the functionality of Vela.
    This can be as simple as Vela crashing upon runtime,
    where an error message is printed out for the user to read.
    However, in certain cases the bug is much harder to spot.
    For example, Vela might run gracefully but optimize a network incorrectly.
    Thus the error will manifest in a silent error, such as incorrect outputs.
    In this case, as much information about the system and settings
    should be noted. This will be used later on when handling the root cause.
    A sufficient documentation of the problem is crucial when dealing with
    these silent errors.

2. Sharing the existence of the bug
    * In order to resolve the issue, its existence needs to be shared
    with the Vela community.  Even though the user could fix it by themselves
    in the first step, it is important that is is shared with the community.
    This step will be detailed in the following sections.

3. The community tackles the problem
    * When the bug has been properly detailed, the Vela community will try to
    find a solution. During this phase, further information may be requested
    or shared through the corresponding bug ticket. If a solution is found,
    the ticket will be updated with the relevant information.

As mentioned, the second step is detailed below.

### Opening the form

To add a new bug report, navigate to the
[Vela issue tracker](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/issues).

Click the button "New issue" or navigate to
[the following page](https://gitlab.arm.com/artificial-intelligence/ethos-u/ethos-u-vela/-/issues/new)

### Title

Write a title that describes the problem in a short and meaningful way.
An example of a good title is "AssertionError in weight_compressor.py
when optimizing DS-CNN". An example of a poor title is "Vela crashes on Linux".

### Type

'Issue' will be preselected and the only option.

### Description

This section should contain as much information as possible,
in order to allow someone else to reproduce the issue.

Please provide the following details:

- Vela command line and options.

- Configuration files. If these contain information that you
do not wish to make public then please upload them using Arm's
[ML Model Review](https://www.arm.com/resources/contact-us/ml-model-review)
process described below.

- Version of Vela used.

- Error messages or other console output from Vela.

- Link to a publicly accessible copy of the input network.
If this is not possible then you will need to upload the network using Arm's
[ML Model Review](https://www.arm.com/resources/contact-us/ml-model-review)
process described below.

- License files related to your network, if applicable.

Attached to the description you can find a toolbox, which contains a button to
upload files. This shall not be used to upload any network files, instead use
[ML Model Review](https://www.arm.com/resources/contact-us/ml-model-review).

### Confidential issues

The Vela project and the issue tracker are public forums.
As such, anyone can see information posted to them.

Thus, any proprietary information you wish to share should be done using Arm's
[ML Model Review](https://www.arm.com/resources/contact-us/ml-model-review)
described below.

Hence, the content of the report should be freely viewable to the public
and thus the "This issue is confidential..." option should not be selected.

### Submitting the report

Submit the report by clicking the "Create issue" button.

A confirmation e-mail will be sent out to the linked e-mail address.
Any further changes to the report will be sent out in the same way.

If you also have a fix for the issue being reported then please follow the
[Vela Contributions Guide](CONTRIBUTING.md) to submit your patch. Also, please
include a link to your merge request in the issue description.

## Arm's [ML Model Review](https://www.arm.com/resources/contact-us/ml-model-review) process.

As mentioned earlier, proprietary or sensitive information should not be shared
on this page.
Instead, any required sensitive content should be sent directly to Arm via
Arm's ML Model Review process.

For instance, if you can't find any publicly hosted examples of your network,
e.g. a network hosted on Model Zoo, upload your own network to
[ML Model Review](https://www.arm.com/resources/contact-us/ml-model-review).

It is a simple process that requires you to complete a short online form
after which you will be sent instructions on how to upload your files.