User Notes
==========
This page will address any notation and units that do not have implied standards in the nuclear data field. Any known bugs, issues, or concerns will also be reported on this page.

Uncommon Notation
=================
* TAZ assumes a penetrability of 1.0 for gamma rays like SAMMY and ATARI. In other words, `<Gg> = 2<gg2>`.

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
- test_wig_sample.TestBayesSample1.test_distributions seems to be failing consistently, but the distribution looks alright.