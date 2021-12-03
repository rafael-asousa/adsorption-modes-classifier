# How did I Make a Scalable Classifier for Molecules using Object Oriented Programming?

One common goal of Quantum Chemistry studies is to understand, at an atomistic level, how two different compounds interact with each other. That was the case of my Master's degree research. I was aiming to understand the interaction of CO2 and other molecules with few atoms clusters of Ni5Ga3, ZrO2, among others. For that purpose, one of the steps was to look into the adsorption modes of the molecule, i.e., I was trying to answer: What are the possible and most probable geometries of a molecule when it interacts with this materials?

Each molecule was used to generate set of about 20 adsorption configurations for each material which yielded a final set of hundreds of configurations to analyze and identify the adsorption modes. How do we approach such an extense? Using machine learning clustering algorithms to automatically identify te adsorption modes and instead of analyzing hundreds of configurations it could be reduced to about 5.

The issue I am willing to talk about on this article appeared when the number of molecules to be analyzed increased from 1 to 8 species. At this point, I was writing code in a function paradigm which would take me to write one different code for each molecule, since the features used for clusterization were different for each molecule and each one would have its own peculiarities. That was a scalability issue.

Then, after studying a lot of Python I saw one way to make my code scalable: Use Object Oriented Programming paradigm so that upon analyzing a new molecule I would only have to write how to obtain the relevant features for it. That saves a lot of time and simplifies the problem, so that even if the code is to be used by another member of the group it would be easily appliable to a different problem.

The basic structure defined for the code is: 

1. Create a class called AdsorptionModesIdentifier that would contain the pipeline for identifying the adsorption modes, which includes collecting data, transforming data and apllying K-Means Clustering Algorithm.
2. Define base classes that would as generalistic as possible to each step of the pipeline, those classes are DataCollector, DataTransformer and DataClassifier.
3. Define problem specific classes, e.g., CO2_collector is a class that collects the data specifically for CO2 molecule.

Python's class heritage allows me to define general procedures of data collection in the DataCollector class and defining only molecule-specific procedures into CO2_collector, which then allows me to easily create new classes as H2_collector, CO_collector, etc. using few lines of code.

Of course, there is a lot of room for improvement on the code, for example one could add another class between CO2_collector and DataCollector to implement reading from different output formats, as I used xyz files but FHI-aims or VASP outputs could be read as well. A class could also be written in order to automatically generate a report of results as representative images of molecules, mean descriptions of each adsorption, etc.