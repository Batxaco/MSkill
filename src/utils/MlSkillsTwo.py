import pickle

import pandas as pd


class Predictor:

    def __init__(self) -> None:
        super().__init__()
        self.instance_data = self.read_instances()
        self.variables_df = self.read_variables()
        self.instances_list = self.transform_instance()
        self.flat_instance = self.concatenation()

    @staticmethod
    def read_instances() -> list:

        with open("../data/data.pickle", "rb") as file:
            instances = pickle.load(file)

        return instances

    @staticmethod
    def read_variables() -> pd.DataFrame:

        with open("../data/variables.pickle", "rb") as file:
            variables = pickle.load(file)

        return pd.DataFrame(variables, columns=('var1', 'var2', 'var3', 'var4', 'var5'))

    def transform_instance(self) -> list:
        instances_list = []
        for instance in range(0, len(self.instance_data[:])):

            elements = []

            for ele in range(len(self.instance_data[instance][0])):
                elements.append("ele" + str(ele))
            total_df = pd.DataFrame(self.instance_data[instance], columns=elements)
            instances_list.append(total_df)

        return instances_list

    def concatenation(self) -> pd.DataFrame:
        instance = []
        for instances_df in range(0, len(self.instances_list)):
            for column in self.instances_list[instances_df].columns:
                instance.append(pd.concat([self.transform_instance()[instances_df][column], self.variables_df], axis=1)
                                .rename(columns={column: 'element', 'var1': 'var1',
                                                 'var2': 'var2', 'var3': 'var3', 'var4': 'var4', 'var5': 'var5'})
                                )

        instance = pd.concat(instance)
        final_instance = instance[instance["element"].str.contains("nan") == False].reset_index().drop("index", axis=1)

        return final_instance
