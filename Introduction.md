# StartUp Name Generator

The idea is that a model generates a very 'techy' name for a future startup.

But what data do I have at hand?
I have the following:
- [Pitchards](https://starthouse.xyz/) -- a list of pitchcards from famous start ups.
- [A kaggle dataset](https://www.kaggle.com/govlab/open-data-500-companies) -- a dataset that represents companies using U.S. government's open data.
- [Y combinator companies](https://data.world/adamhelsinger/y-combinator-companies) 
- [Inc 5000 companies](https://data.world/albert/inc-5000-2016-the-full-list)

Found an extra [dataset](https://www.kaggle.com/theworldbank/doing-business?) that might help with external factors but not sure.

My expectations on how I will be going about to create such a model are:
1. Pull an existing built model that 'understands' sentences or (even better) words.
2. Tune the model to my list of picked up start up names.
3. Hope that it will just generate the rest of a name when you provide a part of it.
    - Also it gotta sound techy as well (e.g. tumblr, pongware, etc.)