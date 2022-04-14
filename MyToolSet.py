import matplotlib.pyplot as pyt 
from pandas.plotting import scatter_matrix
import seaborn as sns
import pandas as pd
import matplotlib.gridspec as gs
import matplotlib.pyplot as plt
import itertools
import numpy as np
# from fancyimpute import MICE as MICE
from scipy import stats as stats
from scipy.special import boxcox1p
from imblearn.combine import SMOTETomek
from collections import defaultdict
from scipy.stats import hmean
from scipy.spatial.distance import cdist
from scipy import stats
import numbers

# Define function for scatter plot
def my_plot(data,target=None):
    
    if target is not None:
        my_hist_by_category(data,target)
        sns.set(style="ticks", color_codes=True)
     
    data=data.select_dtypes(include=np.number)
    if target==None:
        data['idx']='All'
        target='idx'
    sns.pairplot(data, hue=target, palette="husl")
    plt.show()
    if target==None:
        data=data.drop(['idx'])
    return data
###################################################################################################

def my_preprocess(data,scale=None,ignore=[]):
    # Data transformation
    # Convert categorical values to numeric using label encoder
    print('\n Original shape of dataframe----->',data.shape)
    my_missing_values_table(data)
    cols=data.columns.difference(ignore)
    df=pd.DataFrame(data[cols])
    
    df_ignore=pd.DataFrame(data[ignore])
    
    from sklearn.preprocessing import LabelEncoder
    from collections import defaultdict
    d = LabelEncoder()
    print('\n ............................  Initiating feature scaling.....\n')
    numeric_columns=list(df.select_dtypes(include=np.number).columns)
    print('\nColumns that are numerics---->',len(numeric_columns))
    if (str.lower(scale) in ['minmax','standard']) & (len(numeric_columns)!=0):

        df=my_internal_FeatureScale(df,scale)
        print('\n --------------- Scaling of features completed with new dataframe size--------->',df.shape)
        

    # Encoding the categorical variable
    
    
    if  len(numeric_columns)==len(data.columns):
        print('\n No categorical columns found.Exit block.\n')
        
    else:
        print('\n--------------------Initiating categorical encoding--------------------------------\n')
        cols=data.select_dtypes(include='object').columns.difference(ignore)
        print('\nThe columns to be encoded-------',cols)
        for c in cols:
            if c is not None : df[c]=d.fit_transform(data[c])

    print('\n------------Completed Scaling and Encoding-----------------------\n')
    if ignore is not None:
        
        df=pd.concat([df.reset_index(drop=True),df_ignore.reset_index(drop=True)],axis=1)
    print('\n Final dataframe size\n',df.shape)
    my_missing_values_table(df)
    return df



        
    
    
    
    print('\n ---------------------- Shape of dataframe------->',df.shape)
    return df 

# Define function for outlier\boxplot

def my_internal_FeatureScale(df,scale):
    
    scale=str.lower(scale)
    if scale=='minmax':
        from sklearn.preprocessing import MinMaxScaler
        scaler=MinMaxScaler()
        # fit scaler on training data
    if scale=='standard':
        from sklearn.preprocessing import StandardScaler
        scaler=StandardScaler()        
        # transform the training data column
    
    df_new=pd.DataFrame(scaler.fit_transform(df.select_dtypes(np.number)), columns=df.select_dtypes(np.number).columns)
    return df_new
##############################################################################################################################
def my_outlier_flagbyZscore(df):
    data=df.copy(deep=True)
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    data = data.select_dtypes(include=np.number)
    

    print('Original data:'+str(data.shape))
    for col in data.columns:
        my_box_plot(pd.DataFrame(data[col]))
    z = np.abs(stats.zscore(data))
    outlier_index=data[(z > 1.5).all(axis=1)].index
    print('\nRemoval of outliers====',len(outlier_index))
    print('Shape of new dataframe ',data[~data.index.isin(outlier_index)].shape)
    my_box_plot(data[~data.index.isin(outlier_index)])
    return outlier_index,data[~data.index.isin(outlier_index)]
    
def my_box_plot(data):
    
    sns.boxplot(data=data.select_dtypes(np.number), orient="h",linewidth=2.2, palette="Set2")
    plt.show()


# Define function for bar plot by category

def my_hist_by_category(data,target):

    for col in data.select_dtypes(exclude=np.number):
            dframe_temp=pd.DataFrame(data.groupby(col,as_index=False)[target].agg('nunique'))
            sns.barplot(x=col, y=target, data=dframe_temp)
            plt.show()

     
    return


#Define function to impute missing data

def my_impute_values(df):
    #fancy impute removes column names.
    train_cols = df.columns
    # Use MICE to fill in each row's missing features
    df = pd.DataFrame(MICE(verbose=False).complete(df))
    df.columns = train_cols
    return(df)
#####################################################################################################
def my_impute_all(data,ignore=[]):
    df=pd.DataFrame()
    
    data = data.replace(r'^\s+$', np.nan, regex=True)
    categorical_columns = data.select_dtypes(exclude=np.number).columns.difference(ignore)
    numeric_columns = data.select_dtypes(include=np.number).columns.difference(ignore)
    print('Numerics',numeric_columns)
    print('Categories',categorical_columns)
    for ncol in numeric_columns:
        df[ncol]=data[ncol].astype(float)
    for catcol in categorical_columns:
        df[catcol]=data[catcol].astype(str)
    print('\n')
    mis_val_table_ren_columns=my_missing_values_table(df)
    if mis_val_table_ren_columns.shape[0]==0 :
        print('No missing values found in given data')
        return data
    for col in df.columns.difference(ignore):
        print('Treating--> '+col)
        if col not in mis_val_table_ren_columns['Missing Values']:
            print(col+' Has no missing value.Skipping to next one')
            continue
        emptycols= df.columns[df.isna().all()].tolist()
        attributes=list(df.columns.difference(ignore).difference(emptycols))

        if col in attributes: attributes.remove(col)
        
        df[col]=knn_impute(df[col],
                             df[attributes],
                             aggregation_method="mode" if col in categorical_columns else 'median', 
                             k_neighbors=4, 
                             numeric_distance='euclidean',
                             categorical_distance='hamming')
    
    
    
    print(my_missing_values_table(df))
    print("\n --------------------------- The main imputation has been completed. Checking corner cases.\n\n")
    from sklearn.impute import SimpleImputer
    
    
    imp=SimpleImputer(strategy="most_frequent")
    
    for col in df.columns.difference(ignore):
        
        df[col]=imp.fit_transform(df[[col]]).ravel()
          
       
    print('\n Imputation complete.Take a look at the final set.')
    print(my_missing_values_table(df))
    for catcol in categorical_columns:
        df[catcol]=df[catcol].astype('object')
    df.set_index(data.index,inplace=True)
    return(df)
######################################################################################################################
def my_outlier_detection(df):
    #specify  column names to be modelled
    print('\nThis function returns two dataframes with the second one as the treated one.\n')
    print('\n --- Outlier removal process triggered,with shape of data,',df.shape)   
    to_model_columns=df.select_dtypes([np.number]).columns
    from sklearn.ensemble import IsolationForest
    
    clf=IsolationForest(n_estimators=25,
                        max_samples='auto',
                        
                        contamination=float(.05),
                        max_features=1.0,
                        bootstrap=False,
                        n_jobs=-1,
                        random_state=42,
                        verbose=0)
    
    clf.fit(df[to_model_columns])
    pred = clf.predict(df[to_model_columns])
    df['anomaly']=pred
    outliers=df.loc[df['anomaly']==-1]
    outlier_index=list(outliers.index)
    pcnt=100* len(outlier_index)/len(df)
    print("\n----------------------------------- Percentage of data points flagged as outliers ",str(round(pcnt,2))," %---------------")
    #Find the number of anomalies and normal points here points classified -1 are anomalous
    print(df['anomaly'].value_counts())
    
    from sklearn.decomposition import PCA
    pca = PCA(2)
    pca.fit(df[to_model_columns])
    res=pd.DataFrame(pca.transform(df[to_model_columns]))
    Z = np.array(res)
    plt.title("IsolationForest")
    plt.contourf( Z, cmap=plt.cm.Blues_r)
     
    b1 = plt.scatter(res[0], res[1], c='green',
                     s=20,label="normal points")
     
    b1 =plt.scatter(res.iloc[outlier_index,0],res.iloc[outlier_index,1], c='green',s=20,  edgecolor="red",label="predicted outliers")
    plt.legend(loc="upper right")
    plt.show()
    
    return(df,df[df.anomaly==1][df.columns.difference(['anomaly'])])
#########################################################################################################################
def my_featureTransformToNormalDist(column,name,lamb=None):
    print('\nInitiating box cox transformation')
    list_lambda=list()
    
    original=col
    transformed_col, best_lambda = boxcox(df[col],lamb)
    sns.kdeplot(pd.DataFrame([original,transformed_col],columns=['Original','Transformed']),x="Box Cox transformation of "+str(name))
    print("\n Check for normality from the plot and store lambda value to apply to test data later")
    return transformed_col, best_lambda



def my_missing_values_table(df):
        print(df.dtypes)
        mis_val = df.isnull().sum()
        mis_val_percent = 100 * df.isnull().sum() / len(df)
        mis_val_table = pd.concat([mis_val, mis_val_percent], axis=1)
        mis_val_table_ren_columns = mis_val_table.rename(
        columns = {0 : 'Missing Values', 1 : '% of Total Values'})
        mis_val_table_ren_columns = mis_val_table_ren_columns[
            mis_val_table_ren_columns.iloc[:,1] != 0].sort_values(
        '% of Total Values', ascending=False).round(1)
        print ("Your selected dataframe has " + str(df.shape[1]) + " columns.\n"      
            "There are " + str(mis_val_table_ren_columns.shape[0]) +
              " columns that have missing values.")
        return mis_val_table_ren_columns

def my_BalancedSample(df,target,choice=1):
        
        from imblearn.combine import SMOTETomek
        from imblearn.combine import SMOTEENN

        columns=df.columns.difference([target])
        print('\nthe data originally has a shape--------->\n',df[target].value_counts())
        model=SMOTETomek() if choice==1 else SMOTEENN()
        X_smt, y_smt = model.fit_sample(df[columns],df[target])
        X_smt=pd.DataFrame(X_smt, columns=columns)
        X_smt[target]=y_smt
        print('\nthe data now has a shape------->\n',X_smt[target].value_counts())
    

        return(X_smt)


    

def My_PrincipalComponentAnalysis(df,num=None):
    print('\n Principal Component Analysis Triggered')
    df=df.select_dtypes(include=np.number)
    mean_values= np.round(df.mean(axis=0))
    sd_values=np.round(df.std(axis=0))
    if num==None: num=df.shape[1]
    flag=False if (mean_values.max()==0) & (sd_values.min()==1) else True
    if flag:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        print('Data is not scaled. Applying Standard Scaling with mean 0 ans s.d=1')
        df=pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    from sklearn.decomposition import PCA
    pca=PCA(num)
    principalComponents = pca.fit_transform(np.nan_to_num(df))
    
    print(np.round(pca.explained_variance_ratio_, 3))
    sing_vals = np.arange(num) + 1

    fig = plt.figure(figsize=(8,5))
    plt.plot(sing_vals[:12], np.round(pca.explained_variance_ratio_, 3)[:12], 'ro-', linewidth=2)
    plt.title('Scree Plot')
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    leg = plt.legend(['Variance explained'], loc='best', borderpad=0.3, 
                     shadow=False, 
                     markerscale=0.4)
    leg.get_frame().set_alpha(0.4)

    plt.show()
    factor_list=['Component-'+str(idx+1) for idx in range(num)]
    pca_df=pd.DataFrame(pca.components_.T, columns= factor_list,index=df.columns)
    maxLoading = pd.DataFrame(principalComponents,columns=factor_list)
       
    return pca_df,maxLoading


    

# Define function to compare variable distributions between train and test

def my_traintestcomparison(train, test):
    a = len(train.columns)
    if a%2 != 0:
        a += 1
    
    n = np.floor(np.sqrt(a)).astype(np.int64)
    
    while a%n != 0:
        n -= 1
    
    m = (a/n).astype(np.int64)
    coords = list(itertools.product(list(range(m)), list(range(n))))
    
    numerics = train.select_dtypes(include=np.number).columns
    cats = train.select_dtypes(exclude=np.number).columns
    
    fig = plt.figure(figsize=(15, 15))
    axes = gs.GridSpec(m, n)
    axes.update(wspace=0.25, hspace=0.25)
    
    for i in range(len(numerics)):
        x, y = coords[i]
        ax = plt.subplot(axes[x, y])
        col = numerics[i]
        sns.kdeplot(train[col].dropna(), ax=ax, label='train').set(xlabel=col)
        sns.kdeplot(test[col].dropna(), ax=ax, label='test')
        
    for i in range(0, len(cats)):
        x, y = coords[len(numerics)+i]
        ax = plt.subplot(axes[x, y])
        col = cats[i]

        train_temp = train[col].value_counts()
        test_temp = test[col].value_counts()
        train_temp = pd.DataFrame({col: train_temp.index, 'value': train_temp/len(train), 'Set': np.repeat('train', len(train_temp))})
        test_temp = pd.DataFrame({col: test_temp.index, 'value': test_temp/len(test), 'Set': np.repeat('test', len(test_temp))})

        sns.barplot(x=col, y='value', hue='Set', data=pd.concat([train_temp, test_temp]), ax=ax).set(ylabel='Percentage')


   
   

def weighted_hamming(data):
    """ Compute weighted hamming distance on categorical variables. For one variable, it is equal to 1 if
        the values between point A and point B are different, else it is equal the relative frequency of the
        distribution of the value across the variable. For multiple variables, the harmonic mean is computed
        up to a constant factor.
        @params:
            - data = a pandas data frame of categorical variables
        @returns:
            - distance_matrix = a distance matrix with pairwise distance for all attributes
    """
    categories_dist = []
    
    for category in data:
        X = pd.get_dummies(data[category])
        X_mean = X * X.mean()
        X_dot = X_mean.dot(X.transpose())
        X_np = np.asarray(X_dot.replace(0,1,inplace=False))
        categories_dist.append(X_np)
    categories_dist = np.array(categories_dist)
    distances = hmean(categories_dist, axis=0)
    return distances


def distance_matrix(data, numeric_distance = "euclidean", categorical_distance = "jaccard"):
    """ Compute the pairwise distance attribute by attribute in order to account for different variables type:
        - Continuous
        - Categorical
        For ordinal values, provide a numerical representation taking the order into account.
        Categorical variables are transformed into a set of binary ones.
        If both continuous and categorical distance are provided, a Gower-like distance is computed and the numeric
        variables are all normalized in the process.
        If there are missing values, the mean is computed for numerical attributes and the mode for categorical ones.
        
        Note: If weighted-hamming distance is chosen, the computation time increases a lot since it is not coded in C 
        like other distance metrics provided by scipy.
        @params:
            - data                  = pandas dataframe to compute distances on.
            - numeric_distances     = the metric to apply to continuous attributes.
                                      "euclidean" and "cityblock" available.
                                      Default = "euclidean"
            - categorical_distances = the metric to apply to binary attributes.
                                      "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                      available. Default = "jaccard"
        @returns:
            - the distance matrix
    """
    possible_continuous_distances = ["euclidean", "cityblock"]
    possible_binary_distances = ["euclidean", "jaccard", "hamming", "weighted-hamming"]
    number_of_variables = data.shape[1]
    number_of_observations = data.shape[0]

    # Get the type of each attribute (Numeric or categorical)
    is_numeric = [all(isinstance(n, numbers.Number) for n in data.iloc[:, i]) for i, x in enumerate(data)]
    is_all_numeric = sum(is_numeric) == len(is_numeric)
    is_all_categorical = sum(is_numeric) == 0
    is_mixed_type = not is_all_categorical and not is_all_numeric

    # Check the content of the distances parameter
    if numeric_distance not in possible_continuous_distances:
        print("The continuous distance " + numeric_distance + " is not supported.")
        return None
    elif categorical_distance not in possible_binary_distances:
        print("The binary distance " + categorical_distance + " is not supported.")
        return None

    # Separate the data frame into categorical and numeric attributes and normalize numeric data
    if is_mixed_type:
        number_of_numeric_var = sum(is_numeric)
        number_of_categorical_var = number_of_variables - number_of_numeric_var
        data_numeric = data.iloc[:, is_numeric]
        data_numeric = (data_numeric - data_numeric.mean()) / (data_numeric.max() - data_numeric.min())
        data_categorical = data.iloc[:, [not x for x in is_numeric]]

    # Replace missing values with column mean for numeric values and mode for categorical ones. With the mode, it
    # triggers a warning: "SettingWithCopyWarning: A value is trying to be set on a copy of a slice from a DataFrame"
    # but the value are properly replaced
    if is_mixed_type:
        data_numeric.fillna(data_numeric.mean(), inplace=True)
        for x in data_categorical:
            data_categorical[x].fillna(data_categorical[x].mode()[0], inplace=True)
    elif is_all_numeric:
        data.fillna(data.mean(), inplace=True)
    else:
        for x in data:
            data[x].fillna(data[x].mode()[0], inplace=True)

    # "Dummifies" categorical variables in place
    if not is_all_numeric and not (categorical_distance == 'hamming' or categorical_distance == 'weighted-hamming'):
        if is_mixed_type:
            data_categorical = pd.get_dummies(data_categorical)
        else:
            data = pd.get_dummies(data)
    elif not is_all_numeric and categorical_distance == 'hamming':
        if is_mixed_type:
            data_categorical = pd.DataFrame([pd.factorize(data_categorical[x])[0] for x in data_categorical]).transpose()
        else:
            data = pd.DataFrame([pd.factorize(data[x])[0] for x in data]).transpose()

    if is_all_numeric:
        result_matrix = cdist(data, data, metric=numeric_distance)
    elif is_all_categorical:
        if categorical_distance == "weighted-hamming":
            result_matrix = weighted_hamming(data)
        else:
            result_matrix = cdist(data, data, metric=categorical_distance)
    else:
        result_numeric = cdist(data_numeric, data_numeric, metric=numeric_distance)
        if categorical_distance == "weighted-hamming":
            result_categorical = weighted_hamming(data_categorical)
        else:
            result_categorical = cdist(data_categorical, data_categorical, metric=categorical_distance)
        result_matrix = np.array([[1.0*(result_numeric[i, j] * number_of_numeric_var + result_categorical[i, j] *
                               number_of_categorical_var) / number_of_variables for j in range(number_of_observations)] for i in range(number_of_observations)])

    # Fill the diagonal with NaN values
    np.fill_diagonal(result_matrix, np.nan)

    return pd.DataFrame(result_matrix)


def knn_impute(target, attributes, k_neighbors, aggregation_method="mean", numeric_distance="euclidean",
               categorical_distance="jaccard", missing_neighbors_threshold = 0.5):
    """ Replace the missing values within the target variable based on its k nearest neighbors identified with the
        attributes variables. If more than 50% of its neighbors are also missing values, the value is not modified and
        remains missing. If there is a problem in the parameters provided, returns None.
        If to many neighbors also have missing values, leave the missing value of interest unchanged.
        @params:
            - target                        = a vector of n values with missing values that you want to impute. The length has
                                              to be at least n = 3.
            - attributes                    = a data frame of attributes with n rows to match the target variable
            - k_neighbors                   = the number of neighbors to look at to impute the missing values. It has to be a
                                              value between 1 and n.
            - aggregation_method            = how to aggregate the values from the nearest neighbors (mean, median, mode)
                                              Default = "mean"
            - numeric_distances             = the metric to apply to continuous attributes.
                                              "euclidean" and "cityblock" available.
                                              Default = "euclidean"
            - categorical_distances         = the metric to apply to binary attributes.
                                              "jaccard", "hamming", "weighted-hamming" and "euclidean"
                                              available. Default = "jaccard"
            - missing_neighbors_threshold   = minimum of neighbors among the k ones that are not also missing to infer
                                              the correct value. Default = 0.5
        @returns:
            target_completed        = the vector of target values with missing value replaced. If there is a problem
                                      in the parameters, return None
    """

    # Get useful variables
    possible_aggregation_method = ["mean", "median", "mode"]
    number_observations = len(target)
    is_target_numeric = all(isinstance(n, numbers.Number) for n in target)

    # Check for possible errors
    if number_observations < 3:
        print("Not enough observations.")
        return None
    if attributes.shape[0] != number_observations:
        print("The number of observations in the attributes variable is not matching the target variable length.")
        return None
    if k_neighbors > number_observations or k_neighbors < 1:
        print("The range of the number of neighbors is incorrect.")
        return None
    if aggregation_method not in possible_aggregation_method:
        print("The aggregation method is incorrect.")
        return None
    if not is_target_numeric and aggregation_method != "mode":
        print("The only method allowed for categorical target variable is the mode.")
        return None

    # Make sure the data are in the right format
    target = pd.DataFrame(target)
    attributes = pd.DataFrame(attributes)

    # Get the distance matrix and check whether no error was triggered when computing it
    distances = distance_matrix(attributes, numeric_distance, categorical_distance)
    if distances is None:
        return None

    # Get the closest points and compute the correct aggregation method
    for i, value in enumerate(target.iloc[:, 0]):
        if pd.isnull(value):
            order = distances.iloc[i,:].values.argsort()[:k_neighbors]
            closest_to_target = target.iloc[order, :]
            missing_neighbors = [x for x  in closest_to_target.isnull().iloc[:, 0]]
            # Compute the right aggregation method if at least more than 50% of the closest neighbors are not missing
            if sum(missing_neighbors) >= missing_neighbors_threshold * k_neighbors:
                continue
            elif aggregation_method == "mean":
                target.iloc[i] = np.ma.mean(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            elif aggregation_method == "median":
                target.iloc[i] = np.ma.median(np.ma.masked_array(closest_to_target,np.isnan(closest_to_target)))
            else:
                target.iloc[i] = stats.mode(closest_to_target, nan_policy='omit')[0][0]

    return target
   
    
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples,silhouette_score

def my_KMeans(data,n=5,feature_selection=False,fitness_test=False):
        from sklearn.cluster import KMeans
        from sklearn.metrics import silhouette_score,silhouette_samples
        from sklearn.preprocessing import StandardScaler
        from sklearn.metrics import davies_bouldin_score
        
        df=data.select_dtypes(include=np.number)
         
        # 1a)Kmeans: Apply scaling on the feautures usded for clustring. Standard Scaler is used.
        
        mean_values= np.round(df.mean(axis=0))
        sd_values=np.round(df.std(axis=0))
        flag1=1 if (mean_values.max()==0) & (sd_values.min()==1) else 0
        flag2=1 if (sd_values.max(axis=0)==1) & (sd_values.min(axis=0)==0) else 0
             
        if flag1 + flag2==0:
            
            
            scaler = StandardScaler()
            print('Data is not scaled. Applying Standard Scaling.')
            df=pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
            print("\nvalidate mean--")
            print(df.mean(axis=0))
            print('\nvalidate standard deviation-')
            print(df.std(axis=0))
            

        K = range(2,n+1)
        Sum_of_squared_distances = []
        sil=[]
        list_sl=[]
        val=0
        #1b)KMeans: Decide on the best features for Kmeans Clustering
        if feature_selection==True:
                list_bestfeatures_pwc=list()
                for feature in df.columns:
                    
                    dframe_bestfeatures_pwc=pd.DataFrame()
                    for k in K:
                        
                        km = KMeans(n_clusters=k)
                        y_predict= km.fit_predict(pd.DataFrame(df[feature]))
                        silhouette_vals = silhouette_samples(df,y_predict)
                        avg_sil_score=np.mean(silhouette_vals)
                        list_bestfeatures_pwc.append([feature,k,avg_sil_score])
                
                dframe_bestfeatures_pwc=pd.DataFrame(list_bestfeatures_pwc,columns=['FeatureName','Clusters Applied','SilhouetteScore'])
                print('\n------------ Feature Selection Results| Avg. Silhoutte Score -------\n')
                print(pd.pivot_table(dframe_bestfeatures_pwc,index=['FeatureName'],columns=['Clusters Applied'],values='SilhouetteScore'))

                     
                            
        
            
                      
        # 1c)Kmeans: Iterate through different potential values of K to detemine best K by elbow plot visual inspection
        for k in K:
            km = KMeans(n_clusters=k)
            km = km.fit(df)
            labels = km.labels_
            Sum_of_squared_distances.append(km.inertia_)
            sil.append(silhouette_score(df, labels, metric = 'euclidean'))
            
        plt.plot(K, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()
        plt.plot(K, sil, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Silhoutte Score')
        plt.title('Silhoutte Method For Optimal k')
        plt.show()

        y_predict= KMeans(n_clusters=n).fit_predict(df)
        
        # 1d) Kmeans: Fitness test.Optional step of plotting Sillehoute score of each individual sample to inspect quality of cluster formations
        
        if fitness_test==True:
            
            centroids  = km.cluster_centers_
            # get silhouette
            silhouette_vals = silhouette_samples(df,y_predict)
            # silhouette plot
            y_ticks = []
            y_lower = y_upper = 0
            
            print('\n-- Davies_bouldin_score ( closer to zero the better )------> ')
            print(davies_bouldin_score(df,y_predict))

            # Iterate through each cluster.Obtain scores for each sample under a given Cluster.
            
            for i,cluster in enumerate(np.unique(y_predict)):
                
                cluster_silhouette_vals = silhouette_vals[y_predict ==cluster]
                cluster_silhouette_vals.sort()
                y_upper += len(cluster_silhouette_vals)

                plt.barh(range(y_lower,y_upper), cluster_silhouette_vals,height =1);
                plt.text(-0.03,(y_lower+y_upper)/2,str(i+1))
                y_lower += len(cluster_silhouette_vals)

               # Get the average silhouette score 
                avg_score = np.mean(cluster_silhouette_vals)
                list_sl.append(avg_score)
                plt.axvline(avg_score,linestyle ='--',linewidth =2,color = 'green')
                plt.yticks([])
                plt.xlim([-0.1, 1])
                plt.xlabel('Silhouette coefficient values')
                plt.ylabel('Cluster labels')
                
                plt.title('Goodness of fit with frozen value of K')
                plt.show()

                
        if len(list_sl) >0:
                  print('\n Overall goodnes of fit  for the cluster',np.round(100*min(list_sl)))
                  print('******************************************')
                  val=np.round(100*np.mean(list_sl))
        return pd.Series(y_predict),val
   

#########################################################################

{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c4b3ea7c",
   "metadata": {},
   "outputs": [],
   "source": [
    "import  MyToolSet as my_internal_func\n",
    "\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "import warnings\n",
    "warnings.filterwarnings(\"ignore\")\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import precision_score\n",
    "from sklearn.metrics import recall_score\n",
    "from sklearn.metrics import f1_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "5cc2533f",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import data created during the processing stage of Part 1\n",
    "\n",
    "dframe_demo_alltestresults=pd.read_csv('output/dframe_demo_alltestresults.csv')\n",
    "dframe_target=pd.read_csv('T_stage.csv')\n",
    "\n",
    "# The CKD-EPI (Chronic Kidney Disease Epidemiology Collaboration) equation was developed\n",
    "# in an effort to create a more precise formula to estimate glomerular filtrate rate (GFR) from serum creatinine \n",
    "# and other readily available clinical parameters,\n",
    "\n",
    "\n",
    "def eGFScore(row):\n",
    "    k= 0.7 if row['gender']=='Female' else 0.9\n",
    "    alpha=-0.329 if row['gender']=='Female' else -0.411\n",
    "    f1=1.018 if row['gender']=='Female' else 1\n",
    "    f2=1.159 if row['race']=='Black' else 1\n",
    "    \n",
    "    egfr=141* (min(row['Creatinine']/k,1))**alpha * (max(row['Creatinine']/k,1))**(-1.209)*  0.993**row['age']*f1*f2\n",
    "    return round(egfr)\n",
    "         \n",
    "    \n",
    "def CKD_stage(x):\n",
    "    if x>=90: return 1\n",
    "    elif 60<=x<=89: return 2\n",
    "    elif 45<=x<=59: return 3\n",
    "    elif 30<=x<=44 :return 3.5\n",
    "    elif 15<=x<=29: return 4\n",
    "    elif x<=15: return 5\n",
    "    else: return 0"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "3c057a53",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>id</th>\n",
       "      <th>CKD(t=0)</th>\n",
       "      <th>CKD</th>\n",
       "      <th>time</th>\n",
       "      <th>target_new</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1196</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1394</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1254</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1414</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1080</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>295</th>\n",
       "      <td>295</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1157</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>296</th>\n",
       "      <td>296</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1159</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>297</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1008</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>298</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>877</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>299</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>1295</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>300 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  CKD(t=0)  CKD  time  target_new\n",
       "0      0       3.0  3.0  1196           0\n",
       "1      1       4.0  4.0  1394           0\n",
       "2      2       4.0  4.0  1254           0\n",
       "3      3       1.0  3.0  1414           1\n",
       "4      4       3.0  3.0  1080           0\n",
       "..   ...       ...  ...   ...         ...\n",
       "295  295       4.0  4.0  1157           0\n",
       "296  296       4.0  4.0  1159           0\n",
       "297  297       3.0  3.0  1008           0\n",
       "298  298       3.0  3.5   877           1\n",
       "299  299       2.0  2.0  1295           0\n",
       "\n",
       "[300 rows x 5 columns]"
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Treat duplicate EHR records\n",
    "dframe_demo_alltestresults['eGFR']=dframe_demo_alltestresults.apply(eGFScore,axis=1)\n",
    "\n",
    "dframe_demo_alltestresults['CKD']=dframe_demo_alltestresults.apply(lambda x: CKD_stage(x['eGFR']),axis=1)\n",
    "\n",
    "dframe_demo_alltestresults_grp=dframe_demo_alltestresults.groupby('id',as_index=False).agg({'CKD(t=0)':max,'CKD':max,'time':max})\n",
    "dframe_demo_alltestresults_grp['target_new']=dframe_demo_alltestresults_grp.apply(lambda x: 1 if x['CKD']>x['CKD(t=0)'] else 0,axis=1)\n",
    "\n",
    "dframe_demo_alltestresults_grp.groupby('target_new')['id'].nunique()\n",
    "dframe_target=dframe_target.merge(dframe_demo_alltestresults_grp[['id','target_new']],on='id',how='inner')\n",
    "dframe_target.to_csv('T_med_new.csv')\n",
    "dframe_demo_alltestresults_grp"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "6c795166",
   "metadata": {},
   "outputs": [],
   "source": [
    "dframe_medications=pd.read_csv('T_meds.csv')[['id','drug','Medication Duration (days)']]\n",
    "\n",
    "dframe_medications['Medicated']=dframe_medications['Medication Duration (days)'].apply(lambda x: 1 if x>0 else 0)\n",
    "dframe_medications=dframe_medications.groupby(['id','drug'],as_index=False)['Medicated'].max()\n",
    "dframe_medications_statin=dframe_medications[dframe_medications.drug.isin(['atorvastatin','simvastatin'])]\n",
    "dframe_medications_SBP=dframe_medications[dframe_medications.drug.isin(['losartan','metoprolol'])]\n",
    "\n",
    "dframe_medications_statin=dframe_medications_statin.merge(dframe_demo_alltestresults_grp,on='id',how='inner')\n",
    "dframe_medications_SBP=dframe_medications_SBP.merge(dframe_demo_alltestresults_grp,on='id',how='inner')\n",
    "dframe_medications_statin.to_csv('output/Statin.csv')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "082029e5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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eustpk6d2mXlqnNR98gG6EjyvnPPPZeHH36YRx99lDPPPBMzi/kZ5eXlvPXWW8yaNYs777yTiy/uumdIc3Mz3/3ud3nkkUdYtWoVl1xySVwra0tKSjqmJRYVFdHa2hrPj0NEUmTGZ0fws6oGxg9J32cooYc0NDRQXl5O//79efvtt3njjTcALyG2tLQA8PnPf57HH3+cvXv30tjYyGOPPcbnPve5qNc766yzePzxx/ntb3/Lueee2+1nbN++nfb2dr761a+yePFi/vKXvwAwaNAg9uzZA9CRvEeMGMGnn37KI48c6CcRPE9Eep+cLblky5w5c1iyZAnHHnssEydO7CifLFy4kGOPPZbjjz+eBx54gAsvvJCTTjoJ8G6KTp06lU2bNnW5Xnl5OZMmTWLt2rUd58f6jC1btnDRRRd1zI756U9/CsCFF17IpZde2nGz85JLLmHy5MmMHTuWE088seOzwueJSO9i3ZUP0qmqqsrV1NR0OrZu3TqOOuqorMQjHv0eiKTHshVbOHL7s/zjX0fS1rKPp647K6nrmNly51zUe6oquYiIFAgldBGRAqGELiJSIJTQRUQKhBK6iEgGDOpXwo7G9O7GpoQuIpIBp048hJI+RWn9DCX0OGSjDe3u3bu56667Ej5v69atnH322ekMTURylBJ6HLLRhjbZhH7YYYd1Wj0qIr1H7q4U/cMi+HBVaq956GSYe0u3pzQ2NnLOOedQX19PW1sbN9xwA3fffXenNrSXXXYZzz33HOXl5dx8881cffXVvP/++/zyl79k3rx5nHzyySxdupSjj/YajM2aNYuf//zntLW1cdVVV9HU1ES/fv247777mDhxImvWrOGiiy5i//79tLe38+ijj3LDDTfw3nvvMWXKFGbPns2PfvQj5s+fz65du2hpaeEnP/kJ8+fPZ9GiRZ3Ou+yyyzj99NNZvXo1999/P8uWLWPv3r289957nHnmmR2dH0Wk8ORuQs+Sp59+msMOO4z//M//BLy+K3fffXfH642NjcyaNYtbb72VM888kx/84Ac8++yzrF27lgsuuIB58+axYMECHn74YW688UY++OADtm7dygknnMAnn3zCK6+8QnFxMc899xzXXXcdjz76KEuWLOHKK6/k/PPPZ//+/bS1tXHLLbewevXqjpa9ra2tPPbYYwwePJjt27czbdo05s2b1+W8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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "from lifelines import KaplanMeierFitter\n",
    "\n",
    "kmf = KaplanMeierFitter() \n",
    "\n",
    "\n",
    "T = dframe_medications_statin['time']     ## time to event\n",
    "E = dframe_medications_statin['target_new']      ## event occurred or censored\n",
    "\n",
    "\n",
    "groups = dframe_medications_statin['drug']             ## Create the cohorts from the 'Contract' column\n",
    "ix1 = (groups == 'atorvastatin')   ## Cohort 1\n",
    "ix2 = (groups == 'simvastatin')         ## Cohort 2\n",
    "\n",
    "\n",
    "\n",
    "kmf.fit(T[ix1], E[ix1], label='atorvastatin')    ## fit the cohort 1 data\n",
    "ax = kmf.plot()\n",
    "\n",
    "\n",
    "kmf.fit(T[ix2], E[ix2], label='simvastatin')         ## fit the cohort 2 data\n",
    "ax1 = kmf.plot(ax=ax)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "69a412d5",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 1 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "kmf = KaplanMeierFitter() \n",
    "\n",
    "\n",
    "T = dframe_medications_SBP['time']     ## time to event\n",
    "E = dframe_medications_SBP['target_new']      ## event occurred or censored\n",
    "\n",
    "\n",
    "groups = dframe_medications_SBP['drug']             ## Create the cohorts from the 'Contract' column\n",
    "ix1 = (groups == 'losartan')   ## Cohort 1\n",
    "ix2 = (groups == 'metoprolol')         ## Cohort 2\n",
    "\n",
    "\n",
    "\n",
    "kmf.fit(T[ix1], E[ix1], label='losartan')    ## fit the cohort 1 data\n",
    "ax = kmf.plot()\n",
    "\n",
    "\n",
    "kmf.fit(T[ix2], E[ix2], label='metoprolol')         ## fit the cohort 2 data\n",
    "ax1 = kmf.plot(ax=ax)\n",
    "\n"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.8.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
