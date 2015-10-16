# NEEDLE MASTER TOOLS

This project contains the Python and C++ versions of the MATLAB tools I wrote to read Needle Master demonstrations.

To use this, download the game Needle Master [from the Google Play store](https://play.google.com/store/apps/details?id=edu.jhu.lcsr.needlemaster). Go into options and enable data collection. Then, plug your Android device into a computer, go to the storage, and copy out the files from the `needle_master_trials` folder.

The goals of this code are to make sure we generate a reliable set of features across different languages. This version should be more concise and more portable.

Note that the C++ code will only require Boost, so it is fully portable to Windows. That doesn't mean that I will be writing code to support building on Windows immediately, but it should happen eventually.
