User Notes
==========
This page will address any notation and units that do not have implied standards in the nuclear data field. Any known bugs, issues, or concerns will also be reported on this page.

Uncommon Notation
=================
* SAMMY and ATARI assume a penetrability of 1.0 for gamma rays, but FUDGE and TAZ will assume the penetrability is 0.5 so that Gg = &lt;gg2&gt; for infinite degrees of freedom.

Standard Units
==============
TAZ uses the following standard units throughout the code.

| Quantity       | Units |
|:-------------- |:-----:|
| Mass           | amu   |
| Nuclear Radius | fm    |
| Channel Radius | fm    |
| Energy         | eV    |
| Partial Widths | meV   |
| Level Density  | 1/eV  |

**Note:** the units for partial widths can are arbitary as long as it is consistent.

Known Bugs and Issues
=====================
* Need to check if PTBayes.py is normalizing its PT probabilities correctly.