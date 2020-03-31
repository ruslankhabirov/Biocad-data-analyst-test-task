import time
import sys
import math
import pandas as pd
import numpy as np
import scipy
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.inspection import permutation_importance
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import Descriptors
from rdkit.Chem import AllChem
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdFingerprintGenerator
from openbabel import openbabel
from openbabel import pybel


def timer(func):
    """Timer function

    """

    def wrapper(*args):
        start = time.time()
        response = func(*args)
        end = time.time()
        print('Runtime: {} seconds'.format(end - start))

        return response

    return wrapper


def get_morgan_bit_info(smile: str, log_list: list) -> list:
    """Function to get a Morgan bits from smile-string. Contains
    a log for analyzing possible errors
    """
    try:
        molecule = Chem.MolFromSmiles(smile)
        fp = rdMolDescriptors.GetMorganFingerprintAsBitVect(molecule, radius=2,
                                                            nBits=5000)
        legend = [x for x in fp.GetOnBits()]
        return legend
    except Exception as ex:
        log_list.append(ex)
        return []


class CreateSourceDataFrame:
    """Class used for creating the source dataframe

    """

    def __init__(self, smi_table: str, desired_value_table: str):
        """The initial step where we read the two source dataframes

        """
        self.smi_table = pd.read_csv(smi_table, sep=' ', header=None)
        self.smi_table.drop_duplicates(subset=[1], inplace=True)
        self.smi_table.set_index(1, inplace=True)

        self.desired_value_table = pd.read_csv(desired_value_table)
        self.desired_value_table.drop_duplicates(subset='molecule', inplace=True)
        self.desired_value_table.set_index('molecule', inplace=True)
        self.desired_value_table = self.desired_value_table['QPlogS'].to_frame()

    def join_df(self):
        """Step, where we combine the two source dataframes to one

        """
        joined_df = self.smi_table.join(self.desired_value_table)
        joined_df.columns = ['smile', 'QPlogS']
        joined_df.dropna(inplace=True)

        self.smi_table.drop(self.smi_table.index, inplace=True)
        self.desired_value_table.drop(self.desired_value_table.index, inplace=True)

        return joined_df


class MoleculeDynamicsDescriptorClass:
    """Сlass for extracting a molecular dynamics properties of a molecule
    from SMILE-string

    """

    def __init__(self, smile):
        """Initial step, where we convert SMILE-string into OpenBabel mol.
        object

        """
        self.smile = smile
        self.mol = pybel.readstring("smi", smile)

        self.mol.addh()
        self.mol.make2D()

    def get_dipole_moment(self) -> float:
        """Function to extract 2D dipole moment of molecule

        """
        cm = openbabel.OBChargeModel_FindType('mmff94')
        cm.ComputeCharges(self.mol.OBMol)
        dipole = cm.GetDipoleMoment(self.mol.OBMol)
        moment = np.float32(math.sqrt(dipole.GetX() ** 2 + dipole.GetY() ** 2))

        return moment

    def get_mol_weight(self) -> float:
        """Function to get the molecular weight

        """
        weight = np.float32(self.mol.molwt)

        return weight


class CreateMorganBitMatrix:
    """Class where we creates a source data, constructed by
    using Morgan circular algorithm

    """

    def __init__(self, source_df, log_list: list):
        """Initial step, that makes copy of source dataframe

        """
        self.source_df = source_df.copy()
        self.source_df.QPlogS = np.float32(self.source_df.QPlogS)
        source_df.drop(source_df.index, inplace=True)
        self.log_list = log_list

    def create_list_dataframe(self):
        """In this step we create intermediate dataframe with column,
        that contains list of calculated by Morgan algorithm bits

        """
        self.source_df['smile'] = [get_morgan_bit_info(smile, self.log_list) for smile in
                                   self.source_df['smile'].values]
        self.source_df.columns = ['bit_info', 'QPlogS']

    def create_bits_matrix(self):
        """This method converts column with list of Morgan bits into
        a dummy-dataframe by using multilabel binarizer. Returns two
        scipy sparsed matrixes

        """
        mlb = MultiLabelBinarizer()
        bits = self.source_df['bit_info'].to_frame()
        targets = scipy.sparse.csr_matrix(self.source_df['QPlogS'].to_frame().values)

        self.source_df.drop(self.source_df.index, inplace=True)

        bit_df = pd.DataFrame(mlb.fit_transform(bits['bit_info']), columns=mlb.classes_,
                              index=bits.index)
        bit_df = scipy.sparse.csr_matrix(bit_df.values)

        return bit_df, targets


@timer
def create_source_dataframe(smi_table: str, desired_value_table: str):
    """Function, that creates source dataframe. Receives input strings
    with the names of csv-files, provided as a part of the test task.
    Returns source dataframe, that using in a forming of the next
    dataframes

    """
    creator_object = CreateSourceDataFrame(smi_table, desired_value_table)
    source_data = creator_object.join_df()

    return source_data


@timer
def create_dynamics_source_dataframe(source_data):
    """Function, that creates molecule dynamics array

    """
    log_list = []
    source_data = source_data.copy()
    source_data['dipole'] = None
    source_data['MW'] = None
    print("New columns created, waiting for the program ends")

    for index, row in source_data.iterrows():
        try:
            descriptor = MoleculeDynamicsDescriptorClass(row.loc['smile'])
            source_data.loc[index, ['dipole']] = descriptor.get_dipole_moment()
            source_data.loc[index, ['MW']] = descriptor.get_mol_weight()
        except Exception as ex:
            source_data.loc[index, ['dipole']] = None
            source_data.loc[index, ['MW']] = None
            log_list.append(ex)

    dipole_data = pd.DataFrame()
    targets = source_data['QPlogS'].values.reshape(-1, 1)
    dipole_data['dipole'] = source_data['dipole']
    dipole_data['MW'] = source_data['MW']
    dipole_data = dipole_data.values
    print('Array size: ' + str(dipole_data.shape))

    return log_list, dipole_data, targets


@timer
def create_morgan_bits_dataframe(source_data):
    """Function, that creates Morgan bits matrix. PLEASE DON'T
    FORGET TO CONVERT SPARSE MATRIX BACK TO DENSE VERSION BEFORE
    USING IT IN A MODEL

    """
    log_list = []
    print('Source data size: ' + str(source_data.shape[0]) +
          ' rows, ' + str(source_data.shape[1]) + ' columns')
    morgan_creator_object = CreateMorganBitMatrix(source_data, log_list)
    morgan_creator_object.create_list_dataframe()
    bit_data, targets = morgan_creator_object.create_bits_matrix()
    print('Matrix size: ' + str(bit_data.shape))
    print('Memory usage: {} GB'.format((sys.getsizeof(bit_data) +
                                        sys.getsizeof(targets)) * 9.31e-10))

    return log_list, bit_data, targets


@timer
def split_array(X, y, data_split_status):
    """Function, that splits the source arrays into test and train parts.
    Uses a data_split_status flag, that shows to function do we need to
    scale not only the target part of data (y), but also training data too
    (X).

    """
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = MinMaxScaler()
    scaler.fit(y_train)
    y_train = scaler.transform(y_train)
    y_test = scaler.transform(y_test)

    print("X_train array size: {}".format(X_train.shape))
    print("y_train array size: {}".format(y_train.shape))
    print("\ny_train features minimal value: {}".format(y_train.min(axis=0)))
    print("y_train features maximal value: {}".format(y_train.max(axis=0)))
    print("y_test features minimal value: {}".format(y_test.min(axis=0)))
    print("y_test features maximal value: {}".format(y_test.max(axis=0)))

    if data_split_status:
        scaler = MinMaxScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        print("\nX_train features minimal value: {}".format(X_train.min(axis=0)))
        print("X_train features maximal value: {}".format(X_train.max(axis=0)))
        print("X_test features minimal value: {}".format(X_test.min(axis=0)))
        print("X_test features maximal value: {}".format(X_test.max(axis=0)))

    return X_train, X_test, y_train, y_test


@timer
def ridge_regression(X_train, X_test, y_train, y_test):
    """Function to check the efficiency of Ridge regression

    """
    pipe = Pipeline([("ridge", Ridge())])
    param_grid = {'ridge__alpha': [1, 5, 10, 100], 'ridge__max_iter': [100, 1000, 10000]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("The best value of R^2, using a cross-validation: {:.2f}".format(grid.best_score_))
    print("R^2 on test array: {:.2f}".format(grid.score(X_test, y_test)))
    print("Best parameters: {}".format(grid.best_params_))


@timer
def lasso_regression(X_train, X_test, y_train, y_test):
    """Function to check the efficiency of Lasso regression

    """
    pipe = Pipeline([("lasso", Lasso())])
    param_grid = {'lasso__alpha': [0.001, 0.01, 0.1, 1], 'lasso__max_iter': [10, 100, 1000, 10000]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("The best value of R^2, using a cross-validation: {:.2f}".format(grid.best_score_))
    print("R^2 on test array: {:.2f}".format(grid.score(X_test, y_test)))
    print("Best parameters: {}".format(grid.best_params_))


@timer
def svm_regression(X_train, X_test, y_train, y_test):
    """Function to check the efficiency of SVR

    """
    pipe = Pipeline([("svm", SVR())])
    param_grid = {'svm__C': [0.1, 1, 5, 10], 'svm__max_iter': [100, 1000, 10000]}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("The best value of R^2, using a cross-validation: {:.2f}".format(grid.best_score_))
    print("R^2 on test array: {:.2f}".format(grid.score(X_test, y_test)))
    print("Best parameters: {}".format(grid.best_params_))


@timer
def decision_tree_regression(X_train, X_test, y_train, y_test):
    """Function to check the efficiency of DTR

    """
    pipe = Pipeline([('dtr', DecisionTreeRegressor())])
    param_grid = {'dtr__max_depth': [2, 5, 10], 'dtr__criterion': ['mse', 'friedman_mse', 'mae']}
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=5)
    grid.fit(X_train, y_train)

    print("The best value of R^2, using a cross-validation: {:.2f}".format(grid.best_score_))
    print("R^2 on test array: {:.2f}".format(grid.score(X_test, y_test)))
    print("Best parameters: {}".format(grid.best_params_))


@timer
def get_feature_importances(model, X_test, y_test):
    """Function to get the feature importances of a model.
    Receives input fitted on train data model and test arrays.
    Returns a list of importances of every feature.

    """
    importances = permutation_importance(model, X_test, y_test, n_repeats=5, random_state=0)
    importances_list = importances.importances_mean
    importance_series = pd.Series(importances_list)
    scaler = MinMaxScaler()
    importance_array = scaler.fit_transform(importance_series.values.reshape(-1, 1))
    importance_array = [i[0] for i in importance_array.tolist()]

    return importance_array
