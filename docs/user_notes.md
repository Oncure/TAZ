User Notes
==========
This page will address any notation and units that do not have implied standards in the nuclear data field. Any known bugs, issues, or concerns will also be reported on this page.

Uncommon Notation
=================
* TAZ assumes a penetrability of 1.0 for gamma rays like SAMMY and ATARI. In other words, the expectation value of `Gg` is `2&lt;gg2&gt;` for infinite degrees of freedom.

Standard Units
==============
TAZ uses the following standard units throughout the code.

| Quantity       | Units |
|:-------------- |:-----:|
| Mass           | amu   |
| Nuclear Radius | √b    |
| Channel Radius | √b    |
| Energy         | eV    |
| Partial Widths | meV   |
| Level Density  | 1/eV  |

**Note:** the units for partial widths can are arbitary as long as it is consistent.

Known Bugs and Issues
=====================
<<<<<<< HEAD
* Need to check if PTBayes.py is normalizing its PT probabilities correctly.

Things to Change
================
* Match units and assumptions with ATARI.
* Implement better Distributions class and merging protocol.
* Rename `Levels.py` to something more appropriate.
* Overhaul `RunMaster`.
* Implement Verification notebooks as unit tests.
=======
- No known bugs or issues at this time.
>>>>>>> new_runmaster
