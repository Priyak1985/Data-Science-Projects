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
   "cell_type": "markdown",
   "id": "535a847a",
   "metadata": {},
   "source": [
    "# <font color='Blue'> Part 2: Analysis of Chronic Kidney Disease Progression in Patients </font>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "a44bf162",
   "metadata": {},
   "source": [
    "###### Chronic kidney disease,  or CKD, is a condition characterized by a gradual loss of kidney function over time. Early detection can help prevent the progression of kidney disease to kidney failure\n",
    "\n",
    "For this task, you are given a set of longitudinal data (attached) of different lab measurements for patients diagnosed with chronic kidney disease (CKD). Furthermore, you are also given the information whether these patients progress in their CKD stage or not in the future. Using this dataset, you are required to come up with a solution to predict whether a patient will progress in CKD staging given the patient's past longitudinal information."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "c660dd50",
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
   "cell_type": "markdown",
   "id": "02f5e5b0",
   "metadata": {},
   "source": [
    "### 1. Data Import"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "adcbabba",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import data created during the processing stage of Part 1\n",
    "\n",
    "dframe_demoalltestresults=pd.read_csv('output/dframe_demo_alltestresults.csv')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "0bda5f87",
   "metadata": {},
   "outputs": [],
   "source": [
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
    "    egfr=141* (min(row['Creatinine_mean']/k,1))**alpha * (max(row['Creatinine_mean']/k,1))**(-1.209)*  0.993**row['age']*f1*f2\n",
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
    "    else: return 0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "77f3a429",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(14, 1)\n",
      "(300, 1)\n",
      "   id  rankscore\n",
      "0   0        1.0\n",
      "1   0        2.0\n",
      "2   0        3.0\n",
      "3   0        4.0\n",
      "4   0        5.0\n",
      "(4200, 2)\n"
     ]
    },
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
       "      <th>gender</th>\n",
       "      <th>CKD(t=0)</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>SBP</th>\n",
       "      <th>HGB</th>\n",
       "      <th>Lipoprotein</th>\n",
       "      <th>Time Window</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.78</td>\n",
       "      <td>147.5</td>\n",
       "      <td>13.13</td>\n",
       "      <td>157.9</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.78</td>\n",
       "      <td>147.5</td>\n",
       "      <td>13.65</td>\n",
       "      <td>157.9</td>\n",
       "      <td>2.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.78</td>\n",
       "      <td>147.5</td>\n",
       "      <td>12.63</td>\n",
       "      <td>157.9</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.78</td>\n",
       "      <td>147.5</td>\n",
       "      <td>13.36</td>\n",
       "      <td>157.9</td>\n",
       "      <td>4.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>Male</td>\n",
       "      <td>0.0</td>\n",
       "      <td>5.78</td>\n",
       "      <td>147.5</td>\n",
       "      <td>13.53</td>\n",
       "      <td>157.9</td>\n",
       "      <td>5.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id gender  CKD(t=0)  Glucose    SBP    HGB  Lipoprotein  Time Window\n",
       "0   0   Male       0.0     5.78  147.5  13.13        157.9          1.0\n",
       "1   0   Male       0.0     5.78  147.5  13.65        157.9          2.0\n",
       "2   0   Male       0.0     5.78  147.5  12.63        157.9          3.0\n",
       "3   0   Male       0.0     5.78  147.5  13.36        157.9          4.0\n",
       "4   0   Male       0.0     5.78  147.5  13.53        157.9          5.0"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "\n",
    "# Treat duplicate EHR records\n",
    "dframe_demoalltestresults=dframe_demoalltestresults.groupby(['id','gender','CKD(t=0)','time'],as_index=False)[['Glucose','SBP','HGB','Lipoprotein']].mean()\n",
    "dframe_demoalltestresults['rankscore']=dframe_demoalltestresults.sort_values(['id','time']).groupby(['id'],as_index=False)['time'].rank(ascending=False)\n",
    "\n",
    "dframe_demoalltestresults=dframe_demoalltestresults.sort_values(['id','time'])[dframe_demoalltestresults.rankscore<15]\n",
    "\n",
    "\n",
    "dframe_crossjoin1=pd.DataFrame(dframe_demoalltestresults[\"rankscore\"].sort_values().unique())\n",
    "dframe_crossjoin2=pd.DataFrame(dframe_demoalltestresults[\"id\"].sort_values().unique())\n",
    "print(dframe_crossjoin1.shape)\n",
    "print(dframe_crossjoin2.shape)\n",
    "dframe_cros12=dframe_crossjoin2.merge(dframe_crossjoin1,how='cross')\n",
    "dframe_cros12.columns=['id','rankscore']\n",
    "print(dframe_cros12.head(5))\n",
    "\n",
    "print(dframe_cros12.shape)\n",
    "\n",
    "dframe_temporalData=dframe_demoalltestresults.merge(dframe_cros12,how='right',on=['id','rankscore'])\n",
    "dframe_temporalData.update(dframe_temporalData.sort_values(['id','time']).groupby('id',as_index=False).ffill())\n",
    "\n",
    "del(dframe_temporalData['time'])\n",
    "dframe_temporalData.rename(columns={'rankscore':'Time Window'},inplace=True)\n",
    "dframe_smoothened_temporal=dframe_temporalData\n",
    "dframe_temporalData.head()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "6eb04506",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(300, 57)\n"
     ]
    },
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
       "      <th>gender</th>\n",
       "      <th>CKD(t=0)</th>\n",
       "      <th>Glucose_1.0</th>\n",
       "      <th>Glucose_2.0</th>\n",
       "      <th>Glucose_3.0</th>\n",
       "      <th>Glucose_4.0</th>\n",
       "      <th>Glucose_5.0</th>\n",
       "      <th>Glucose_6.0</th>\n",
       "      <th>Glucose_7.0</th>\n",
       "      <th>...</th>\n",
       "      <th>Lipoprotein_6.0</th>\n",
       "      <th>Lipoprotein_7.0</th>\n",
       "      <th>Lipoprotein_8.0</th>\n",
       "      <th>Lipoprotein_9.0</th>\n",
       "      <th>Lipoprotein_10.0</th>\n",
       "      <th>Lipoprotein_11.0</th>\n",
       "      <th>Lipoprotein_12.0</th>\n",
       "      <th>Lipoprotein_13.0</th>\n",
       "      <th>Lipoprotein_14.0</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.663496</td>\n",
       "      <td>-0.663496</td>\n",
       "      <td>...</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.476681</td>\n",
       "      <td>-0.476681</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>0.436677</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>...</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.893048</td>\n",
       "      <td>-0.893048</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>...</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>0.850895</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>1.946977</td>\n",
       "      <td>1.946977</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.520037</td>\n",
       "      <td>-0.520037</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-0.547628</td>\n",
       "      <td>-0.547628</td>\n",
       "      <td>-0.547628</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>295</th>\n",
       "      <td>295</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.266654</td>\n",
       "      <td>-0.266654</td>\n",
       "      <td>-0.961127</td>\n",
       "      <td>...</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>1.252930</td>\n",
       "      <td>1.252930</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>296</th>\n",
       "      <td>296</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>...</td>\n",
       "      <td>0.189078</td>\n",
       "      <td>0.189078</td>\n",
       "      <td>0.607954</td>\n",
       "      <td>0.607954</td>\n",
       "      <td>1.483688</td>\n",
       "      <td>1.483688</td>\n",
       "      <td>1.483688</td>\n",
       "      <td>0.162921</td>\n",
       "      <td>0.162921</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>297</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>0.756452</td>\n",
       "      <td>0.756452</td>\n",
       "      <td>...</td>\n",
       "      <td>0.805030</td>\n",
       "      <td>0.805030</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>298</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.781308</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.356643</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>-0.613200</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>299</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.475247</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>300 rows ?? 60 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  gender  CKD(t=0)  Glucose_1.0  Glucose_2.0  Glucose_3.0  \\\n",
       "0      0       1       3.0    -0.564285    -0.564285    -0.564285   \n",
       "1      1       0       4.0     1.915973     1.915973     1.915973   \n",
       "2      2       0       4.0     0.216996     0.216996     0.216996   \n",
       "3      3       1       1.0    -0.570486    -0.570486    -0.570486   \n",
       "4      4       0       3.0     2.672452     2.672452     2.672452   \n",
       "..   ...     ...       ...          ...          ...          ...   \n",
       "295  295       0       4.0    -0.520881    -0.520881    -0.520881   \n",
       "296  296       0       4.0     0.725449     0.725449     0.725449   \n",
       "297  297       1       3.0     1.791960     1.791960     1.791960   \n",
       "298  298       0       3.0    -0.440273    -0.440273    -0.440273   \n",
       "299  299       1       2.0    -0.303858    -0.303858    -0.303858   \n",
       "\n",
       "     Glucose_4.0  Glucose_5.0  Glucose_6.0  Glucose_7.0  ...  Lipoprotein_6.0  \\\n",
       "0      -0.564285    -0.564285    -0.663496    -0.663496  ...         2.556857   \n",
       "1       1.915973     1.915973     1.915973     1.915973  ...        -0.476681   \n",
       "2       0.216996     0.216996    -0.973528    -0.973528  ...         0.021384   \n",
       "3      -0.570486    -0.570486    -0.570486    -0.570486  ...         0.533065   \n",
       "4       2.672452     2.672452     1.946977     1.946977  ...        -0.520037   \n",
       "..           ...          ...          ...          ...  ...              ...   \n",
       "295    -0.520881    -0.266654    -0.266654    -0.961127  ...         0.746266   \n",
       "296     0.725449     0.725449     0.725449     0.725449  ...         0.189078   \n",
       "297     1.791960     1.791960     0.756452     0.756452  ...         0.805030   \n",
       "298    -0.837114    -0.837114    -0.837114    -0.781308  ...        -0.356643   \n",
       "299    -0.341062    -0.341062    -0.341062    -0.341062  ...        -0.475247   \n",
       "\n",
       "     Lipoprotein_7.0  Lipoprotein_8.0  Lipoprotein_9.0  Lipoprotein_10.0  \\\n",
       "0           2.556857         2.556857         2.556857          2.556857   \n",
       "1          -0.476681        -0.269572        -0.269572         -0.269572   \n",
       "2           0.021384        -0.338727        -0.338727         -0.338727   \n",
       "3           0.533065         0.533065         0.533065          0.533065   \n",
       "4          -0.520037        -1.144231        -1.144231         -1.144231   \n",
       "..               ...              ...              ...               ...   \n",
       "295         0.746266         0.746266         0.746266          0.746266   \n",
       "296         0.189078         0.607954         0.607954          1.483688   \n",
       "297         0.805030         1.547111         1.547111          1.547111   \n",
       "298         0.597205         0.597205         0.597205          0.597205   \n",
       "299        -0.719980        -0.719980        -0.719980         -0.719980   \n",
       "\n",
       "     Lipoprotein_11.0  Lipoprotein_12.0  Lipoprotein_13.0  Lipoprotein_14.0  \\\n",
       "0            0.890310          0.890310          0.890310          0.890310   \n",
       "1           -0.536878         -0.536878         -0.536878          0.436677   \n",
       "2           -0.338727         -0.893048         -0.893048          0.021384   \n",
       "3           -0.205791         -0.205791         -0.205791          0.850895   \n",
       "4           -1.144231         -0.547628         -0.547628         -0.547628   \n",
       "..                ...               ...               ...               ...   \n",
       "295          0.746266          0.746266          1.252930          1.252930   \n",
       "296          1.483688          1.483688          0.162921          0.162921   \n",
       "297          1.547111          1.504829          1.504829          1.504829   \n",
       "298          0.597205          0.597205          0.597205         -0.613200   \n",
       "299         -0.719980         -0.719980         -0.719980         -0.719980   \n",
       "\n",
       "     target  \n",
       "0         1  \n",
       "1         0  \n",
       "2         1  \n",
       "3         0  \n",
       "4         1  \n",
       "..      ...  \n",
       "295       1  \n",
       "296       0  \n",
       "297       1  \n",
       "298       0  \n",
       "299       0  \n",
       "\n",
       "[300 rows x 60 columns]"
      ]
     },
     "execution_count": 5,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from sklearn.preprocessing import StandardScaler\n",
    "dframe_container=pd.DataFrame()\n",
    "dframe_container['id']=dframe_temporalData['id'].unique()\n",
    "\n",
    "scaler=StandardScaler()        \n",
    "    \n",
    "for col in ['Glucose','HGB','SBP','Lipoprotein']:\n",
    "    dframe_smoothened_temporal[col]=scaler.fit_transform(dframe_smoothened_temporal[col].values.astype(float).reshape(-1, 1))\n",
    "    dframe_temp=pd.pivot_table(dframe_smoothened_temporal,\n",
    "                               index=['id'],\n",
    "                               columns='Time Window',\n",
    "                               values=col,\n",
    "                               aggfunc=max)\n",
    "    orig_col=dframe_temp.columns\n",
    "    new_col=[ col+\"_\"+str(x) for x in orig_col]\n",
    "    dframe_temp.columns=new_col\n",
    "    dframe_temp=dframe_temp.reset_index()\n",
    "    \n",
    "    dframe_container=dframe_container.merge(dframe_temp,how='left',on='id')\n",
    "print(dframe_container.shape)    \n",
    "dframe_container=pd.read_csv('Output/Prediction_Ready.csv')[['id','gender','CKD(t=0)']].merge(dframe_container,on='id',how='inner')\n",
    "dframe_container=dframe_container.merge(pd.read_csv('T_Stage.csv')[['id','target']],on='id',how='inner')\n",
    "dframe_container['gender']=dframe_container['gender'].apply(lambda x: 1 if x==\"Male\" else 0)\n",
    "# dframe_container=dframe_container.set_index('id')\n",
    "dframe_container\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "879586c4",
   "metadata": {
    "scrolled": true
   },
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
       "      <th>gender</th>\n",
       "      <th>CKD(t=0)</th>\n",
       "      <th>Glucose_1.0</th>\n",
       "      <th>Glucose_2.0</th>\n",
       "      <th>Glucose_3.0</th>\n",
       "      <th>Glucose_4.0</th>\n",
       "      <th>Glucose_5.0</th>\n",
       "      <th>Glucose_6.0</th>\n",
       "      <th>Glucose_7.0</th>\n",
       "      <th>...</th>\n",
       "      <th>Lipoprotein_6.0</th>\n",
       "      <th>Lipoprotein_7.0</th>\n",
       "      <th>Lipoprotein_8.0</th>\n",
       "      <th>Lipoprotein_9.0</th>\n",
       "      <th>Lipoprotein_10.0</th>\n",
       "      <th>Lipoprotein_11.0</th>\n",
       "      <th>Lipoprotein_12.0</th>\n",
       "      <th>Lipoprotein_13.0</th>\n",
       "      <th>Lipoprotein_14.0</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.663496</td>\n",
       "      <td>-0.663496</td>\n",
       "      <td>...</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.476681</td>\n",
       "      <td>-0.476681</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>0.436677</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>...</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.893048</td>\n",
       "      <td>-0.893048</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>...</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>0.850895</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>2.672452</td>\n",
       "      <td>1.946977</td>\n",
       "      <td>1.946977</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.520037</td>\n",
       "      <td>-0.520037</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-1.144231</td>\n",
       "      <td>-0.547628</td>\n",
       "      <td>-0.547628</td>\n",
       "      <td>-0.547628</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>295</th>\n",
       "      <td>295</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.266654</td>\n",
       "      <td>-0.266654</td>\n",
       "      <td>-0.961127</td>\n",
       "      <td>...</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>1.252930</td>\n",
       "      <td>1.252930</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>296</th>\n",
       "      <td>296</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>...</td>\n",
       "      <td>0.189078</td>\n",
       "      <td>0.189078</td>\n",
       "      <td>0.607954</td>\n",
       "      <td>0.607954</td>\n",
       "      <td>1.483688</td>\n",
       "      <td>1.483688</td>\n",
       "      <td>1.483688</td>\n",
       "      <td>0.162921</td>\n",
       "      <td>0.162921</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>297</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>0.756452</td>\n",
       "      <td>0.756452</td>\n",
       "      <td>...</td>\n",
       "      <td>0.805030</td>\n",
       "      <td>0.805030</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>298</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.781308</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.356643</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>-0.613200</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>299</td>\n",
       "      <td>1</td>\n",
       "      <td>2.0</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.475247</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>300 rows ?? 60 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  gender  CKD(t=0)  Glucose_1.0  Glucose_2.0  Glucose_3.0  \\\n",
       "0      0       1       3.0    -0.564285    -0.564285    -0.564285   \n",
       "1      1       0       4.0     1.915973     1.915973     1.915973   \n",
       "2      2       0       4.0     0.216996     0.216996     0.216996   \n",
       "3      3       1       1.0    -0.570486    -0.570486    -0.570486   \n",
       "4      4       0       3.0     2.672452     2.672452     2.672452   \n",
       "..   ...     ...       ...          ...          ...          ...   \n",
       "295  295       0       4.0    -0.520881    -0.520881    -0.520881   \n",
       "296  296       0       4.0     0.725449     0.725449     0.725449   \n",
       "297  297       1       3.0     1.791960     1.791960     1.791960   \n",
       "298  298       0       3.0    -0.440273    -0.440273    -0.440273   \n",
       "299  299       1       2.0    -0.303858    -0.303858    -0.303858   \n",
       "\n",
       "     Glucose_4.0  Glucose_5.0  Glucose_6.0  Glucose_7.0  ...  Lipoprotein_6.0  \\\n",
       "0      -0.564285    -0.564285    -0.663496    -0.663496  ...         2.556857   \n",
       "1       1.915973     1.915973     1.915973     1.915973  ...        -0.476681   \n",
       "2       0.216996     0.216996    -0.973528    -0.973528  ...         0.021384   \n",
       "3      -0.570486    -0.570486    -0.570486    -0.570486  ...         0.533065   \n",
       "4       2.672452     2.672452     1.946977     1.946977  ...        -0.520037   \n",
       "..           ...          ...          ...          ...  ...              ...   \n",
       "295    -0.520881    -0.266654    -0.266654    -0.961127  ...         0.746266   \n",
       "296     0.725449     0.725449     0.725449     0.725449  ...         0.189078   \n",
       "297     1.791960     1.791960     0.756452     0.756452  ...         0.805030   \n",
       "298    -0.837114    -0.837114    -0.837114    -0.781308  ...        -0.356643   \n",
       "299    -0.341062    -0.341062    -0.341062    -0.341062  ...        -0.475247   \n",
       "\n",
       "     Lipoprotein_7.0  Lipoprotein_8.0  Lipoprotein_9.0  Lipoprotein_10.0  \\\n",
       "0           2.556857         2.556857         2.556857          2.556857   \n",
       "1          -0.476681        -0.269572        -0.269572         -0.269572   \n",
       "2           0.021384        -0.338727        -0.338727         -0.338727   \n",
       "3           0.533065         0.533065         0.533065          0.533065   \n",
       "4          -0.520037        -1.144231        -1.144231         -1.144231   \n",
       "..               ...              ...              ...               ...   \n",
       "295         0.746266         0.746266         0.746266          0.746266   \n",
       "296         0.189078         0.607954         0.607954          1.483688   \n",
       "297         0.805030         1.547111         1.547111          1.547111   \n",
       "298         0.597205         0.597205         0.597205          0.597205   \n",
       "299        -0.719980        -0.719980        -0.719980         -0.719980   \n",
       "\n",
       "     Lipoprotein_11.0  Lipoprotein_12.0  Lipoprotein_13.0  Lipoprotein_14.0  \\\n",
       "0            0.890310          0.890310          0.890310          0.890310   \n",
       "1           -0.536878         -0.536878         -0.536878          0.436677   \n",
       "2           -0.338727         -0.893048         -0.893048          0.021384   \n",
       "3           -0.205791         -0.205791         -0.205791          0.850895   \n",
       "4           -1.144231         -0.547628         -0.547628         -0.547628   \n",
       "..                ...               ...               ...               ...   \n",
       "295          0.746266          0.746266          1.252930          1.252930   \n",
       "296          1.483688          1.483688          0.162921          0.162921   \n",
       "297          1.547111          1.504829          1.504829          1.504829   \n",
       "298          0.597205          0.597205          0.597205         -0.613200   \n",
       "299         -0.719980         -0.719980         -0.719980         -0.719980   \n",
       "\n",
       "     target  \n",
       "0         1  \n",
       "1         0  \n",
       "2         1  \n",
       "3         0  \n",
       "4         1  \n",
       "..      ...  \n",
       "295       1  \n",
       "296       0  \n",
       "297       1  \n",
       "298       0  \n",
       "299       0  \n",
       "\n",
       "[300 rows x 60 columns]"
      ]
     },
     "execution_count": 6,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Scaling of data\n",
    "dframe_temporal_final=dframe_container\n",
    "# dframe_temporal_final=my_internal_func.my_preprocess(dframe_container,'standard',ignore=['id','target','CKD(t=0)'])\n",
    "dframe_temporal_final.to_csv('output/dframe_temporal_final.csv',index=False)\n",
    "dframe_temporal_final"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ae32b100",
   "metadata": {},
   "source": [
    "#### <font color='grey'> Advanced Approach(Deep Learning)</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ab06c4fb",
   "metadata": {},
   "source": [
    "The recall value of the linear model(s) are quite poor. Let us try to improve it using the identified features but with the longitudinal data nature intact. We will build a RNN. But before that we need to standardise the observation period across patients to some extent. By visual inspection of the file 'Window.csv' produced by the code we can come to a cutoff period."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "1d1bc235",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(170, 60) (130, 60)\n",
      "(170, 56)\n",
      "(130, 56)\n"
     ]
    },
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
       "      <th>Glucose_1.0</th>\n",
       "      <th>Glucose_2.0</th>\n",
       "      <th>Glucose_3.0</th>\n",
       "      <th>Glucose_4.0</th>\n",
       "      <th>Glucose_5.0</th>\n",
       "      <th>Glucose_6.0</th>\n",
       "      <th>Glucose_7.0</th>\n",
       "      <th>Glucose_8.0</th>\n",
       "      <th>Glucose_9.0</th>\n",
       "      <th>Glucose_10.0</th>\n",
       "      <th>...</th>\n",
       "      <th>Lipoprotein_5.0</th>\n",
       "      <th>Lipoprotein_6.0</th>\n",
       "      <th>Lipoprotein_7.0</th>\n",
       "      <th>Lipoprotein_8.0</th>\n",
       "      <th>Lipoprotein_9.0</th>\n",
       "      <th>Lipoprotein_10.0</th>\n",
       "      <th>Lipoprotein_11.0</th>\n",
       "      <th>Lipoprotein_12.0</th>\n",
       "      <th>Lipoprotein_13.0</th>\n",
       "      <th>Lipoprotein_14.0</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.564285</td>\n",
       "      <td>-0.663496</td>\n",
       "      <td>-0.663496</td>\n",
       "      <td>-0.663496</td>\n",
       "      <td>0.123986</td>\n",
       "      <td>0.123986</td>\n",
       "      <td>...</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>2.556857</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "      <td>0.890310</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>0.216996</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>-0.973528</td>\n",
       "      <td>...</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>0.021384</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.338727</td>\n",
       "      <td>-0.893048</td>\n",
       "      <td>-0.893048</td>\n",
       "      <td>0.021384</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.570486</td>\n",
       "      <td>-0.793709</td>\n",
       "      <td>0.458821</td>\n",
       "      <td>0.458821</td>\n",
       "      <td>...</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>0.533065</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>-0.205791</td>\n",
       "      <td>0.850895</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>-0.012428</td>\n",
       "      <td>-0.012428</td>\n",
       "      <td>-0.012428</td>\n",
       "      <td>-0.012428</td>\n",
       "      <td>-0.012428</td>\n",
       "      <td>-0.012428</td>\n",
       "      <td>0.465022</td>\n",
       "      <td>0.465022</td>\n",
       "      <td>0.465022</td>\n",
       "      <td>1.345514</td>\n",
       "      <td>...</td>\n",
       "      <td>-1.745133</td>\n",
       "      <td>-1.745133</td>\n",
       "      <td>-1.745133</td>\n",
       "      <td>-1.360298</td>\n",
       "      <td>-1.360298</td>\n",
       "      <td>-1.360298</td>\n",
       "      <td>-1.454894</td>\n",
       "      <td>-1.454894</td>\n",
       "      <td>-1.454894</td>\n",
       "      <td>-1.243127</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>-0.725502</td>\n",
       "      <td>-0.725502</td>\n",
       "      <td>-0.725502</td>\n",
       "      <td>-0.725502</td>\n",
       "      <td>-0.725502</td>\n",
       "      <td>-0.725502</td>\n",
       "      <td>-0.725502</td>\n",
       "      <td>-0.595289</td>\n",
       "      <td>-0.595289</td>\n",
       "      <td>-0.595289</td>\n",
       "      <td>...</td>\n",
       "      <td>2.020810</td>\n",
       "      <td>2.020810</td>\n",
       "      <td>2.020810</td>\n",
       "      <td>2.020810</td>\n",
       "      <td>2.020810</td>\n",
       "      <td>1.642782</td>\n",
       "      <td>1.642782</td>\n",
       "      <td>1.642782</td>\n",
       "      <td>1.642782</td>\n",
       "      <td>1.642782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>292</th>\n",
       "      <td>0.861863</td>\n",
       "      <td>0.861863</td>\n",
       "      <td>0.861863</td>\n",
       "      <td>0.861863</td>\n",
       "      <td>0.861863</td>\n",
       "      <td>0.644841</td>\n",
       "      <td>0.644841</td>\n",
       "      <td>0.644841</td>\n",
       "      <td>0.644841</td>\n",
       "      <td>0.644841</td>\n",
       "      <td>...</td>\n",
       "      <td>0.221685</td>\n",
       "      <td>0.153963</td>\n",
       "      <td>0.153963</td>\n",
       "      <td>0.153963</td>\n",
       "      <td>-0.021614</td>\n",
       "      <td>-0.021614</td>\n",
       "      <td>0.441694</td>\n",
       "      <td>0.441694</td>\n",
       "      <td>0.441694</td>\n",
       "      <td>0.441694</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>295</th>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.520881</td>\n",
       "      <td>-0.266654</td>\n",
       "      <td>-0.266654</td>\n",
       "      <td>-0.961127</td>\n",
       "      <td>-0.961127</td>\n",
       "      <td>-0.961127</td>\n",
       "      <td>-0.961127</td>\n",
       "      <td>...</td>\n",
       "      <td>1.333194</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>0.746266</td>\n",
       "      <td>1.252930</td>\n",
       "      <td>1.252930</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>1.791960</td>\n",
       "      <td>0.756452</td>\n",
       "      <td>0.756452</td>\n",
       "      <td>0.756452</td>\n",
       "      <td>0.458821</td>\n",
       "      <td>0.458821</td>\n",
       "      <td>...</td>\n",
       "      <td>0.805030</td>\n",
       "      <td>0.805030</td>\n",
       "      <td>0.805030</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.547111</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1.504829</td>\n",
       "      <td>1.504829</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.440273</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.837114</td>\n",
       "      <td>-0.781308</td>\n",
       "      <td>-0.781308</td>\n",
       "      <td>-0.781308</td>\n",
       "      <td>-0.272855</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.356643</td>\n",
       "      <td>-0.356643</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>0.597205</td>\n",
       "      <td>-0.613200</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.303858</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.341062</td>\n",
       "      <td>-0.675897</td>\n",
       "      <td>-0.675897</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.475247</td>\n",
       "      <td>-0.475247</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "      <td>-0.719980</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>170 rows ?? 56 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Glucose_1.0  Glucose_2.0  Glucose_3.0  Glucose_4.0  Glucose_5.0  \\\n",
       "0      -0.564285    -0.564285    -0.564285    -0.564285    -0.564285   \n",
       "2       0.216996     0.216996     0.216996     0.216996     0.216996   \n",
       "3      -0.570486    -0.570486    -0.570486    -0.570486    -0.570486   \n",
       "6      -0.012428    -0.012428    -0.012428    -0.012428    -0.012428   \n",
       "8      -0.725502    -0.725502    -0.725502    -0.725502    -0.725502   \n",
       "..           ...          ...          ...          ...          ...   \n",
       "292     0.861863     0.861863     0.861863     0.861863     0.861863   \n",
       "295    -0.520881    -0.520881    -0.520881    -0.520881    -0.266654   \n",
       "297     1.791960     1.791960     1.791960     1.791960     1.791960   \n",
       "298    -0.440273    -0.440273    -0.440273    -0.837114    -0.837114   \n",
       "299    -0.303858    -0.303858    -0.303858    -0.341062    -0.341062   \n",
       "\n",
       "     Glucose_6.0  Glucose_7.0  Glucose_8.0  Glucose_9.0  Glucose_10.0  ...  \\\n",
       "0      -0.663496    -0.663496    -0.663496     0.123986      0.123986  ...   \n",
       "2      -0.973528    -0.973528    -0.973528    -0.973528     -0.973528  ...   \n",
       "3      -0.570486    -0.570486    -0.793709     0.458821      0.458821  ...   \n",
       "6      -0.012428     0.465022     0.465022     0.465022      1.345514  ...   \n",
       "8      -0.725502    -0.725502    -0.595289    -0.595289     -0.595289  ...   \n",
       "..           ...          ...          ...          ...           ...  ...   \n",
       "292     0.644841     0.644841     0.644841     0.644841      0.644841  ...   \n",
       "295    -0.266654    -0.961127    -0.961127    -0.961127     -0.961127  ...   \n",
       "297     0.756452     0.756452     0.756452     0.458821      0.458821  ...   \n",
       "298    -0.837114    -0.781308    -0.781308    -0.781308     -0.272855  ...   \n",
       "299    -0.341062    -0.341062    -0.341062    -0.675897     -0.675897  ...   \n",
       "\n",
       "     Lipoprotein_5.0  Lipoprotein_6.0  Lipoprotein_7.0  Lipoprotein_8.0  \\\n",
       "0           2.556857         2.556857         2.556857         2.556857   \n",
       "2           0.021384         0.021384         0.021384        -0.338727   \n",
       "3           0.533065         0.533065         0.533065         0.533065   \n",
       "6          -1.745133        -1.745133        -1.745133        -1.360298   \n",
       "8           2.020810         2.020810         2.020810         2.020810   \n",
       "..               ...              ...              ...              ...   \n",
       "292         0.221685         0.153963         0.153963         0.153963   \n",
       "295         1.333194         0.746266         0.746266         0.746266   \n",
       "297         0.805030         0.805030         0.805030         1.547111   \n",
       "298        -0.356643        -0.356643         0.597205         0.597205   \n",
       "299        -0.475247        -0.475247        -0.719980        -0.719980   \n",
       "\n",
       "     Lipoprotein_9.0  Lipoprotein_10.0  Lipoprotein_11.0  Lipoprotein_12.0  \\\n",
       "0           2.556857          2.556857          0.890310          0.890310   \n",
       "2          -0.338727         -0.338727         -0.338727         -0.893048   \n",
       "3           0.533065          0.533065         -0.205791         -0.205791   \n",
       "6          -1.360298         -1.360298         -1.454894         -1.454894   \n",
       "8           2.020810          1.642782          1.642782          1.642782   \n",
       "..               ...               ...               ...               ...   \n",
       "292        -0.021614         -0.021614          0.441694          0.441694   \n",
       "295         0.746266          0.746266          0.746266          0.746266   \n",
       "297         1.547111          1.547111          1.547111          1.504829   \n",
       "298         0.597205          0.597205          0.597205          0.597205   \n",
       "299        -0.719980         -0.719980         -0.719980         -0.719980   \n",
       "\n",
       "     Lipoprotein_13.0  Lipoprotein_14.0  \n",
       "0            0.890310          0.890310  \n",
       "2           -0.893048          0.021384  \n",
       "3           -0.205791          0.850895  \n",
       "6           -1.454894         -1.243127  \n",
       "8            1.642782          1.642782  \n",
       "..                ...               ...  \n",
       "292          0.441694          0.441694  \n",
       "295          1.252930          1.252930  \n",
       "297          1.504829          1.504829  \n",
       "298          0.597205         -0.613200  \n",
       "299         -0.719980         -0.719980  \n",
       "\n",
       "[170 rows x 56 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Train Test split with imbalanced retained .\n",
    "\n",
    "minority_id=dframe_temporal_final[dframe_temporal_final.target==1]['id'].unique()\n",
    "majority_id=dframe_temporal_final[dframe_temporal_final.target==0]['id'].unique()\n",
    "len(minority_id)\n",
    "\n",
    "minority_test_id=pd.Series(minority_id).sample(frac=0.10,random_state=0)\n",
    "majority_test_id=pd.Series(majority_id).sample(frac=0.60,random_state=0)\n",
    "\n",
    "dframe_test=dframe_temporal_final[dframe_temporal_final.id.isin(minority_test_id) | dframe_temporal_final.id.isin(majority_test_id)]\n",
    "dframe_train=dframe_temporal_final[~dframe_temporal_final.id.isin(dframe_test.id)]\n",
    "print(dframe_train.shape,dframe_test.shape)\n",
    "\n",
    "\n",
    "dframe_train_X_time=dframe_train.filter(like=\"_\")\n",
    "dframe_train_static=dframe_train[['gender','CKD(t=0)']]\n",
    "dframe_train_Y=dframe_train['target']\n",
    "print(dframe_train_X_time.shape)\n",
    "\n",
    "dframe_test_X_time=dframe_test.filter(like=\"_\")\n",
    "dframe_test_static=dframe_test[['gender','CKD(t=0)']]\n",
    "dframe_test_Y=dframe_test['target']\n",
    "print(dframe_test_X_time.shape)\n",
    "\n",
    "dframe_train_X_time"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ac4af12e",
   "metadata": {},
   "source": [
    "### RNN with LSTM layer"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "6c091b23",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras.models import Sequential\n",
    "# Import `Dense` from `keras.layers`\n",
    "from keras.layers import Dense, Dropout\n",
    "from keras.layers import LSTM\n",
    "from keras.layers import Input\n",
    "from keras.models import Model\n",
    "import keras\n",
    "import tensorflow\n",
    "\n",
    "\n",
    "# tensorflow.random.set_seed(0)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "bc36d63e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(170, 14, 4)"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# reshape input to be 3D [samples, timesteps, features]\n",
    "dframe_train_X_time = dframe_train_X_time.to_numpy().reshape((dframe_train_X_time.shape[0], 14, 4))\n",
    "dframe_test_X_time=dframe_test_X_time.to_numpy().reshape((dframe_test_X_time.shape[0], 14, 4))\n",
    "dframe_train_X_time.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "b0a0a2af",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Citation from \n",
    "import keras.backend as K\n",
    "from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint\n",
    "from tensorflow.keras.optimizers import RMSprop\n",
    "from keras.callbacks import EarlyStopping\n",
    "\n",
    "main_input = Input(shape=(dframe_train_X_time.shape[1], dframe_train_X_time.shape[2]), name='main_input')\n",
    "\n",
    "lstm_out = LSTM(16, dropout=0.25, recurrent_dropout=0.2)(main_input)\n",
    "\n",
    "auxiliary_output = Dense(1, activation='sigmoid', name='aux_output')(lstm_out)\n",
    "\n",
    "auxiliary_input = Input(shape=(2,), name='aux_input')\n",
    "\n",
    "x = keras.layers.concatenate([lstm_out, auxiliary_input])\n",
    "\n",
    "\n",
    "x = Dense(4, activation='relu')(x)\n",
    "x= Dropout(0.25)(x)\n",
    "\n",
    "\n",
    "\n",
    "# And finally we add the main logistic regression layer\n",
    "main_output = Dense(1, activation='sigmoid', name='main_output')(x)\n",
    "#This defines a model with two inputs and two outputs:\n",
    "model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "3448dbc9",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"model\"\n",
      "__________________________________________________________________________________________________\n",
      " Layer (type)                   Output Shape         Param #     Connected to                     \n",
      "==================================================================================================\n",
      " main_input (InputLayer)        [(None, 14, 4)]      0           []                               \n",
      "                                                                                                  \n",
      " lstm (LSTM)                    (None, 16)           1344        ['main_input[0][0]']             \n",
      "                                                                                                  \n",
      " aux_input (InputLayer)         [(None, 2)]          0           []                               \n",
      "                                                                                                  \n",
      " concatenate (Concatenate)      (None, 18)           0           ['lstm[0][0]',                   \n",
      "                                                                  'aux_input[0][0]']              \n",
      "                                                                                                  \n",
      " dense (Dense)                  (None, 4)            76          ['concatenate[0][0]']            \n",
      "                                                                                                  \n",
      " dropout (Dropout)              (None, 4)            0           ['dense[0][0]']                  \n",
      "                                                                                                  \n",
      " main_output (Dense)            (None, 1)            5           ['dropout[0][0]']                \n",
      "                                                                                                  \n",
      " aux_output (Dense)             (None, 1)            17          ['lstm[0][0]']                   \n",
      "                                                                                                  \n",
      "==================================================================================================\n",
      "Total params: 1,442\n",
      "Trainable params: 1,442\n",
      "Non-trainable params: 0\n",
      "__________________________________________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "577686ac",
   "metadata": {},
   "outputs": [],
   "source": [
    "def recall_m(y_true, y_pred):\n",
    "    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))\n",
    "    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))\n",
    "    recall = true_positives / (possible_positives + K.epsilon())\n",
    "    return recall\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "2899394b",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam',\n",
    "              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},loss_weights={'main_output': 1., 'aux_output': 0.10},\n",
    "              metrics=['acc',recall_m])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b289e096",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/80\n",
      "1/6 [====>.........................] - ETA: 9s - loss: 1.0785 - main_output_loss: 1.0123 - aux_output_loss: 0.6617 - main_output_acc: 0.6562 - main_output_recall_m: 1.0000 - aux_output_acc: 0.6250 - aux_output_recall_m: 0.7500WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 2s 94ms/step - loss: 1.2891 - main_output_loss: 1.2206 - aux_output_loss: 0.6854 - main_output_acc: 0.5280 - main_output_recall_m: 0.9449 - aux_output_acc: 0.5342 - aux_output_recall_m: 0.5036 - val_loss: 1.3344 - val_main_output_loss: 1.2573 - val_aux_output_loss: 0.7704 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.2222 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 2/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 1.4539 - main_output_loss: 1.3821 - aux_output_loss: 0.7182 - main_output_acc: 0.3438 - main_output_recall_m: 0.7857 - aux_output_acc: 0.4375 - aux_output_recall_m: 0.5000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 10ms/step - loss: 1.2311 - main_output_loss: 1.1622 - aux_output_loss: 0.6885 - main_output_acc: 0.4907 - main_output_recall_m: 0.9170 - aux_output_acc: 0.5528 - aux_output_recall_m: 0.4808 - val_loss: 1.2933 - val_main_output_loss: 1.2171 - val_aux_output_loss: 0.7619 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.2222 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 3/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 1.0193 - main_output_loss: 0.9499 - aux_output_loss: 0.6937 - main_output_acc: 0.6250 - main_output_recall_m: 0.9524 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.6190WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 11ms/step - loss: 1.2105 - main_output_loss: 1.1407 - aux_output_loss: 0.6987 - main_output_acc: 0.5404 - main_output_recall_m: 0.7924 - aux_output_acc: 0.4845 - aux_output_recall_m: 0.4305 - val_loss: 1.2550 - val_main_output_loss: 1.1796 - val_aux_output_loss: 0.7538 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.1111 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 4/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 1.0069 - main_output_loss: 0.9357 - aux_output_loss: 0.7120 - main_output_acc: 0.5938 - main_output_recall_m: 0.9444 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.5000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 11ms/step - loss: 1.1176 - main_output_loss: 1.0488 - aux_output_loss: 0.6885 - main_output_acc: 0.5342 - main_output_recall_m: 0.9389 - aux_output_acc: 0.5528 - aux_output_recall_m: 0.6697 - val_loss: 1.2163 - val_main_output_loss: 1.1417 - val_aux_output_loss: 0.7459 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.2222 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 5/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.9968 - main_output_loss: 0.9272 - aux_output_loss: 0.6965 - main_output_acc: 0.6250 - main_output_recall_m: 0.9000 - aux_output_acc: 0.4375 - aux_output_recall_m: 0.4500WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 11ms/step - loss: 1.1691 - main_output_loss: 1.0998 - aux_output_loss: 0.6930 - main_output_acc: 0.5093 - main_output_recall_m: 0.7526 - aux_output_acc: 0.4534 - aux_output_recall_m: 0.3553 - val_loss: 1.1828 - val_main_output_loss: 1.1088 - val_aux_output_loss: 0.7397 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.1111 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 6/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 1.3861 - main_output_loss: 1.3168 - aux_output_loss: 0.6927 - main_output_acc: 0.3750 - main_output_recall_m: 0.8000 - aux_output_acc: 0.5625 - aux_output_recall_m: 0.8000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 1.0316 - main_output_loss: 0.9636 - aux_output_loss: 0.6800 - main_output_acc: 0.5528 - main_output_recall_m: 0.7924 - aux_output_acc: 0.5466 - aux_output_recall_m: 0.4450 - val_loss: 1.1516 - val_main_output_loss: 1.0780 - val_aux_output_loss: 0.7359 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.1111 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 7/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 1.0733 - main_output_loss: 1.0068 - aux_output_loss: 0.6657 - main_output_acc: 0.5625 - main_output_recall_m: 1.0000 - aux_output_acc: 0.6562 - aux_output_recall_m: 0.5385WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 1.0670 - main_output_loss: 0.9988 - aux_output_loss: 0.6823 - main_output_acc: 0.5404 - main_output_recall_m: 0.9165 - aux_output_acc: 0.5963 - aux_output_recall_m: 0.6085 - val_loss: 1.1245 - val_main_output_loss: 1.0513 - val_aux_output_loss: 0.7319 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.1111 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 8/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8079 - main_output_loss: 0.7387 - aux_output_loss: 0.6929 - main_output_acc: 0.5938 - main_output_recall_m: 0.9500 - aux_output_acc: 0.5625 - aux_output_recall_m: 0.4500WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 1.0299 - main_output_loss: 0.9614 - aux_output_loss: 0.6845 - main_output_acc: 0.5466 - main_output_recall_m: 0.9519 - aux_output_acc: 0.5963 - aux_output_recall_m: 0.6015 - val_loss: 1.1003 - val_main_output_loss: 1.0273 - val_aux_output_loss: 0.7298 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.1111 - val_aux_output_recall_m: 0.0000e+00\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 9/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 1.1078 - main_output_loss: 1.0369 - aux_output_loss: 0.7086 - main_output_acc: 0.4688 - main_output_recall_m: 0.7778 - aux_output_acc: 0.3750 - aux_output_recall_m: 0.3333WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 1.0361 - main_output_loss: 0.9660 - aux_output_loss: 0.7010 - main_output_acc: 0.5404 - main_output_recall_m: 0.7615 - aux_output_acc: 0.4845 - aux_output_recall_m: 0.3500 - val_loss: 1.0752 - val_main_output_loss: 1.0026 - val_aux_output_loss: 0.7255 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.1111 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 10/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.9457 - main_output_loss: 0.8787 - aux_output_loss: 0.6695 - main_output_acc: 0.5312 - main_output_recall_m: 0.8947 - aux_output_acc: 0.6562 - aux_output_recall_m: 0.6316WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 1.0799 - main_output_loss: 1.0119 - aux_output_loss: 0.6802 - main_output_acc: 0.4907 - main_output_recall_m: 0.7364 - aux_output_acc: 0.5776 - aux_output_recall_m: 0.4082 - val_loss: 1.0424 - val_main_output_loss: 0.9704 - val_aux_output_loss: 0.7197 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.2222 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 11/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.9839 - main_output_loss: 0.9146 - aux_output_loss: 0.6936 - main_output_acc: 0.6250 - main_output_recall_m: 1.0000 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.4706WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.9619 - main_output_loss: 0.8933 - aux_output_loss: 0.6860 - main_output_acc: 0.5217 - main_output_recall_m: 0.7112 - aux_output_acc: 0.5528 - aux_output_recall_m: 0.3863 - val_loss: 1.0073 - val_main_output_loss: 0.9359 - val_aux_output_loss: 0.7141 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 12/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 1.1103 - main_output_loss: 1.0414 - aux_output_loss: 0.6889 - main_output_acc: 0.4375 - main_output_recall_m: 0.8000 - aux_output_acc: 0.5625 - aux_output_recall_m: 0.4000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 1.0014 - main_output_loss: 0.9327 - aux_output_loss: 0.6863 - main_output_acc: 0.5093 - main_output_recall_m: 0.7282 - aux_output_acc: 0.5652 - aux_output_recall_m: 0.3344 - val_loss: 0.9772 - val_main_output_loss: 0.9061 - val_aux_output_loss: 0.7109 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 13/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8993 - main_output_loss: 0.8287 - aux_output_loss: 0.7062 - main_output_acc: 0.5000 - main_output_recall_m: 0.8333 - aux_output_acc: 0.4375 - aux_output_recall_m: 0.2222WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.9515 - main_output_loss: 0.8828 - aux_output_loss: 0.6875 - main_output_acc: 0.5031 - main_output_recall_m: 0.7361 - aux_output_acc: 0.5155 - aux_output_recall_m: 0.2318 - val_loss: 0.9521 - val_main_output_loss: 0.8810 - val_aux_output_loss: 0.7109 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 14/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8013 - main_output_loss: 0.7309 - aux_output_loss: 0.7036 - main_output_acc: 0.6562 - main_output_recall_m: 1.0000 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.2105WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.9236 - main_output_loss: 0.8547 - aux_output_loss: 0.6890 - main_output_acc: 0.5217 - main_output_recall_m: 0.9003 - aux_output_acc: 0.5466 - aux_output_recall_m: 0.1789 - val_loss: 0.9190 - val_main_output_loss: 0.8483 - val_aux_output_loss: 0.7071 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 15/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.9185 - main_output_loss: 0.8503 - aux_output_loss: 0.6825 - main_output_acc: 0.3125 - main_output_recall_m: 0.6000 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.3333WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.8353 - main_output_loss: 0.7660 - aux_output_loss: 0.6933 - main_output_acc: 0.5155 - main_output_recall_m: 0.6850 - aux_output_acc: 0.5466 - aux_output_recall_m: 0.2411 - val_loss: 0.8920 - val_main_output_loss: 0.8217 - val_aux_output_loss: 0.7030 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 16/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8920 - main_output_loss: 0.8216 - aux_output_loss: 0.7049 - main_output_acc: 0.5000 - main_output_recall_m: 0.9375 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.1250WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.8343 - main_output_loss: 0.7647 - aux_output_loss: 0.6964 - main_output_acc: 0.5528 - main_output_recall_m: 0.8900 - aux_output_acc: 0.5155 - aux_output_recall_m: 0.1502 - val_loss: 0.8657 - val_main_output_loss: 0.7957 - val_aux_output_loss: 0.7000 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 17/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8235 - main_output_loss: 0.7526 - aux_output_loss: 0.7086 - main_output_acc: 0.5000 - main_output_recall_m: 0.7778 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.0556WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.8261 - main_output_loss: 0.7562 - aux_output_loss: 0.6987 - main_output_acc: 0.5031 - main_output_recall_m: 0.6653 - aux_output_acc: 0.5217 - aux_output_recall_m: 0.1323 - val_loss: 0.8481 - val_main_output_loss: 0.7781 - val_aux_output_loss: 0.6999 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 18/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8671 - main_output_loss: 0.7961 - aux_output_loss: 0.7097 - main_output_acc: 0.4688 - main_output_recall_m: 0.7222 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.1667WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.8125 - main_output_loss: 0.7422 - aux_output_loss: 0.7027 - main_output_acc: 0.4907 - main_output_recall_m: 0.7973 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.0990 - val_loss: 0.8350 - val_main_output_loss: 0.7647 - val_aux_output_loss: 0.7034 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 1.0000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 19/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8252 - main_output_loss: 0.7552 - aux_output_loss: 0.7002 - main_output_acc: 0.4375 - main_output_recall_m: 0.6667 - aux_output_acc: 0.5625 - aux_output_recall_m: 0.2222WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.8290 - main_output_loss: 0.7597 - aux_output_loss: 0.6929 - main_output_acc: 0.4658 - main_output_recall_m: 0.6238 - aux_output_acc: 0.5280 - aux_output_recall_m: 0.1421 - val_loss: 0.8290 - val_main_output_loss: 0.7582 - val_aux_output_loss: 0.7075 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 20/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7971 - main_output_loss: 0.7274 - aux_output_loss: 0.6968 - main_output_acc: 0.4688 - main_output_recall_m: 0.6000 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.0000e+00WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7978 - main_output_loss: 0.7272 - aux_output_loss: 0.7061 - main_output_acc: 0.4907 - main_output_recall_m: 0.7456 - aux_output_acc: 0.5093 - aux_output_recall_m: 0.0935 - val_loss: 0.8197 - val_main_output_loss: 0.7487 - val_aux_output_loss: 0.7104 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 21/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7014 - main_output_loss: 0.6277 - aux_output_loss: 0.7365 - main_output_acc: 0.6250 - main_output_recall_m: 0.6842 - aux_output_acc: 0.4375 - aux_output_recall_m: 0.0526WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7685 - main_output_loss: 0.6985 - aux_output_loss: 0.7000 - main_output_acc: 0.5342 - main_output_recall_m: 0.5600 - aux_output_acc: 0.5280 - aux_output_recall_m: 0.1194 - val_loss: 0.8139 - val_main_output_loss: 0.7429 - val_aux_output_loss: 0.7107 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 22/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7588 - main_output_loss: 0.6888 - aux_output_loss: 0.6998 - main_output_acc: 0.5938 - main_output_recall_m: 0.5882 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.0588WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.8104 - main_output_loss: 0.7400 - aux_output_loss: 0.7038 - main_output_acc: 0.5217 - main_output_recall_m: 0.7123 - aux_output_acc: 0.4845 - aux_output_recall_m: 0.0782 - val_loss: 0.8068 - val_main_output_loss: 0.7359 - val_aux_output_loss: 0.7091 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 23/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.8122 - main_output_loss: 0.7447 - aux_output_loss: 0.6752 - main_output_acc: 0.4062 - main_output_recall_m: 0.4286 - aux_output_acc: 0.5625 - aux_output_recall_m: 0.1429WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.8073 - main_output_loss: 0.7367 - aux_output_loss: 0.7056 - main_output_acc: 0.4472 - main_output_recall_m: 0.4687 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.1064 - val_loss: 0.7965 - val_main_output_loss: 0.7260 - val_aux_output_loss: 0.7051 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 24/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7389 - main_output_loss: 0.6670 - aux_output_loss: 0.7196 - main_output_acc: 0.5625 - main_output_recall_m: 0.6000 - aux_output_acc: 0.3750 - aux_output_recall_m: 0.0000e+00WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7713 - main_output_loss: 0.7016 - aux_output_loss: 0.6968 - main_output_acc: 0.5404 - main_output_recall_m: 0.5286 - aux_output_acc: 0.5155 - aux_output_recall_m: 0.1190 - val_loss: 0.7894 - val_main_output_loss: 0.7189 - val_aux_output_loss: 0.7042 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 25/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7611 - main_output_loss: 0.6923 - aux_output_loss: 0.6879 - main_output_acc: 0.4688 - main_output_recall_m: 0.4706 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.1176WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7713 - main_output_loss: 0.7001 - aux_output_loss: 0.7115 - main_output_acc: 0.4658 - main_output_recall_m: 0.5811 - aux_output_acc: 0.4783 - aux_output_recall_m: 0.0873 - val_loss: 0.7851 - val_main_output_loss: 0.7145 - val_aux_output_loss: 0.7053 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 26/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7547 - main_output_loss: 0.6830 - aux_output_loss: 0.7177 - main_output_acc: 0.6875 - main_output_recall_m: 0.7059 - aux_output_acc: 0.4062 - aux_output_recall_m: 0.0000e+00WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7741 - main_output_loss: 0.7030 - aux_output_loss: 0.7110 - main_output_acc: 0.4845 - main_output_recall_m: 0.3947 - aux_output_acc: 0.4969 - aux_output_recall_m: 0.0875 - val_loss: 0.7843 - val_main_output_loss: 0.7135 - val_aux_output_loss: 0.7081 - val_main_output_acc: 0.4444 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 27/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7691 - main_output_loss: 0.6985 - aux_output_loss: 0.7059 - main_output_acc: 0.5000 - main_output_recall_m: 0.4706 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.0588WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7726 - main_output_loss: 0.7013 - aux_output_loss: 0.7124 - main_output_acc: 0.4472 - main_output_recall_m: 0.3268 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.0819 - val_loss: 0.7833 - val_main_output_loss: 0.7123 - val_aux_output_loss: 0.7100 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 28/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7735 - main_output_loss: 0.7014 - aux_output_loss: 0.7210 - main_output_acc: 0.4062 - main_output_recall_m: 0.2105 - aux_output_acc: 0.3750 - aux_output_recall_m: 0.0000e+00WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7628 - main_output_loss: 0.6919 - aux_output_loss: 0.7094 - main_output_acc: 0.4969 - main_output_recall_m: 0.2854 - aux_output_acc: 0.4783 - aux_output_recall_m: 0.0614 - val_loss: 0.7817 - val_main_output_loss: 0.7106 - val_aux_output_loss: 0.7112 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 29/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7566 - main_output_loss: 0.6831 - aux_output_loss: 0.7342 - main_output_acc: 0.5938 - main_output_recall_m: 0.5789 - aux_output_acc: 0.3438 - aux_output_recall_m: 0.0000e+00WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7639 - main_output_loss: 0.6943 - aux_output_loss: 0.6958 - main_output_acc: 0.5093 - main_output_recall_m: 0.3530 - aux_output_acc: 0.4845 - aux_output_recall_m: 0.0953 - val_loss: 0.7774 - val_main_output_loss: 0.7065 - val_aux_output_loss: 0.7090 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 30/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7398 - main_output_loss: 0.6700 - aux_output_loss: 0.6975 - main_output_acc: 0.5000 - main_output_recall_m: 0.3684 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.1053WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 14ms/step - loss: 0.7689 - main_output_loss: 0.6979 - aux_output_loss: 0.7096 - main_output_acc: 0.4969 - main_output_recall_m: 0.2868 - aux_output_acc: 0.4845 - aux_output_recall_m: 0.0792 - val_loss: 0.7693 - val_main_output_loss: 0.6991 - val_aux_output_loss: 0.7020 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 31/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7410 - main_output_loss: 0.6694 - aux_output_loss: 0.7161 - main_output_acc: 0.5312 - main_output_recall_m: 0.3500 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.1500WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7460 - main_output_loss: 0.6760 - aux_output_loss: 0.7001 - main_output_acc: 0.5528 - main_output_recall_m: 0.2887 - aux_output_acc: 0.5155 - aux_output_recall_m: 0.0919 - val_loss: 0.7625 - val_main_output_loss: 0.6930 - val_aux_output_loss: 0.6947 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 32/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7659 - main_output_loss: 0.6950 - aux_output_loss: 0.7086 - main_output_acc: 0.5312 - main_output_recall_m: 0.2353 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.0588WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7626 - main_output_loss: 0.6914 - aux_output_loss: 0.7113 - main_output_acc: 0.5342 - main_output_recall_m: 0.2265 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.0768 - val_loss: 0.7583 - val_main_output_loss: 0.6894 - val_aux_output_loss: 0.6894 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 33/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7646 - main_output_loss: 0.6914 - aux_output_loss: 0.7321 - main_output_acc: 0.5000 - main_output_recall_m: 0.1176 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.0588WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7584 - main_output_loss: 0.6870 - aux_output_loss: 0.7144 - main_output_acc: 0.5280 - main_output_recall_m: 0.2105 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.0786 - val_loss: 0.7561 - val_main_output_loss: 0.6875 - val_aux_output_loss: 0.6857 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 34/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7476 - main_output_loss: 0.6759 - aux_output_loss: 0.7165 - main_output_acc: 0.4688 - main_output_recall_m: 0.2000 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.1500WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7613 - main_output_loss: 0.6912 - aux_output_loss: 0.7008 - main_output_acc: 0.4969 - main_output_recall_m: 0.1916 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.0870 - val_loss: 0.7540 - val_main_output_loss: 0.6855 - val_aux_output_loss: 0.6842 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 35/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7846 - main_output_loss: 0.7145 - aux_output_loss: 0.7005 - main_output_acc: 0.4062 - main_output_recall_m: 0.1053 - aux_output_acc: 0.4375 - aux_output_recall_m: 0.1053WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7619 - main_output_loss: 0.6919 - aux_output_loss: 0.7001 - main_output_acc: 0.5404 - main_output_recall_m: 0.1851 - aux_output_acc: 0.5093 - aux_output_recall_m: 0.1054 - val_loss: 0.7540 - val_main_output_loss: 0.6856 - val_aux_output_loss: 0.6847 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 36/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7776 - main_output_loss: 0.7089 - aux_output_loss: 0.6862 - main_output_acc: 0.5000 - main_output_recall_m: 0.0714 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.0714WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7680 - main_output_loss: 0.6975 - aux_output_loss: 0.7051 - main_output_acc: 0.4720 - main_output_recall_m: 0.1171 - aux_output_acc: 0.5155 - aux_output_recall_m: 0.1161 - val_loss: 0.7529 - val_main_output_loss: 0.6844 - val_aux_output_loss: 0.6855 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 37/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7493 - main_output_loss: 0.6811 - aux_output_loss: 0.6822 - main_output_acc: 0.5625 - main_output_recall_m: 0.1765 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.1765WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7694 - main_output_loss: 0.6984 - aux_output_loss: 0.7100 - main_output_acc: 0.4845 - main_output_recall_m: 0.1473 - aux_output_acc: 0.4969 - aux_output_recall_m: 0.1235 - val_loss: 0.7519 - val_main_output_loss: 0.6833 - val_aux_output_loss: 0.6858 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 38/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7699 - main_output_loss: 0.6955 - aux_output_loss: 0.7435 - main_output_acc: 0.4375 - main_output_recall_m: 0.1579 - aux_output_acc: 0.3750 - aux_output_recall_m: 0.0526WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7616 - main_output_loss: 0.6913 - aux_output_loss: 0.7025 - main_output_acc: 0.4720 - main_output_recall_m: 0.1199 - aux_output_acc: 0.4969 - aux_output_recall_m: 0.0849 - val_loss: 0.7491 - val_main_output_loss: 0.6806 - val_aux_output_loss: 0.6845 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 39/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7472 - main_output_loss: 0.6778 - aux_output_loss: 0.6937 - main_output_acc: 0.5938 - main_output_recall_m: 0.2000 - aux_output_acc: 0.5625 - aux_output_recall_m: 0.0667WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7581 - main_output_loss: 0.6871 - aux_output_loss: 0.7098 - main_output_acc: 0.5217 - main_output_recall_m: 0.1417 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.0671 - val_loss: 0.7461 - val_main_output_loss: 0.6777 - val_aux_output_loss: 0.6838 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 40/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7578 - main_output_loss: 0.6904 - aux_output_loss: 0.6742 - main_output_acc: 0.5312 - main_output_recall_m: 0.1250 - aux_output_acc: 0.5625 - aux_output_recall_m: 0.1875WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7653 - main_output_loss: 0.6952 - aux_output_loss: 0.7012 - main_output_acc: 0.4720 - main_output_recall_m: 0.1133 - aux_output_acc: 0.4907 - aux_output_recall_m: 0.0979 - val_loss: 0.7437 - val_main_output_loss: 0.6754 - val_aux_output_loss: 0.6825 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 41/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7624 - main_output_loss: 0.6907 - aux_output_loss: 0.7178 - main_output_acc: 0.4375 - main_output_recall_m: 0.1667 - aux_output_acc: 0.4375 - aux_output_recall_m: 0.1667WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7612 - main_output_loss: 0.6908 - aux_output_loss: 0.7049 - main_output_acc: 0.4907 - main_output_recall_m: 0.1508 - aux_output_acc: 0.4596 - aux_output_recall_m: 0.0847 - val_loss: 0.7407 - val_main_output_loss: 0.6726 - val_aux_output_loss: 0.6809 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.0000e+00\n",
      "Epoch 42/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7584 - main_output_loss: 0.6848 - aux_output_loss: 0.7360 - main_output_acc: 0.5000 - main_output_recall_m: 0.2632 - aux_output_acc: 0.4375 - aux_output_recall_m: 0.2105WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7594 - main_output_loss: 0.6893 - aux_output_loss: 0.7008 - main_output_acc: 0.5155 - main_output_recall_m: 0.1743 - aux_output_acc: 0.5031 - aux_output_recall_m: 0.1118 - val_loss: 0.7383 - val_main_output_loss: 0.6702 - val_aux_output_loss: 0.6807 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.2500 - val_aux_output_acc: 0.6667 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 43/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7562 - main_output_loss: 0.6867 - aux_output_loss: 0.6947 - main_output_acc: 0.4688 - main_output_recall_m: 0.1500 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.2000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7584 - main_output_loss: 0.6892 - aux_output_loss: 0.6921 - main_output_acc: 0.5155 - main_output_recall_m: 0.2807 - aux_output_acc: 0.5404 - aux_output_recall_m: 0.3024 - val_loss: 0.7364 - val_main_output_loss: 0.6683 - val_aux_output_loss: 0.6812 - val_main_output_acc: 0.7778 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 44/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7436 - main_output_loss: 0.6755 - aux_output_loss: 0.6814 - main_output_acc: 0.5938 - main_output_recall_m: 0.2000 - aux_output_acc: 0.5938 - aux_output_recall_m: 0.2000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7612 - main_output_loss: 0.6903 - aux_output_loss: 0.7091 - main_output_acc: 0.4969 - main_output_recall_m: 0.1300 - aux_output_acc: 0.4783 - aux_output_recall_m: 0.1208 - val_loss: 0.7391 - val_main_output_loss: 0.6706 - val_aux_output_loss: 0.6854 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 45/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7336 - main_output_loss: 0.6675 - aux_output_loss: 0.6609 - main_output_acc: 0.6875 - main_output_recall_m: 0.5000 - aux_output_acc: 0.6875 - aux_output_recall_m: 0.5000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7469 - main_output_loss: 0.6778 - aux_output_loss: 0.6909 - main_output_acc: 0.5776 - main_output_recall_m: 0.2661 - aux_output_acc: 0.5528 - aux_output_recall_m: 0.1973 - val_loss: 0.7415 - val_main_output_loss: 0.6727 - val_aux_output_loss: 0.6877 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 46/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7678 - main_output_loss: 0.6953 - aux_output_loss: 0.7250 - main_output_acc: 0.5312 - main_output_recall_m: 0.2667 - aux_output_acc: 0.3750 - aux_output_recall_m: 0.0667WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7657 - main_output_loss: 0.6952 - aux_output_loss: 0.7050 - main_output_acc: 0.4783 - main_output_recall_m: 0.1482 - aux_output_acc: 0.4783 - aux_output_recall_m: 0.1431 - val_loss: 0.7426 - val_main_output_loss: 0.6736 - val_aux_output_loss: 0.6898 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 47/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7468 - main_output_loss: 0.6771 - aux_output_loss: 0.6966 - main_output_acc: 0.5938 - main_output_recall_m: 0.4737 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.2632WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7499 - main_output_loss: 0.6806 - aux_output_loss: 0.6936 - main_output_acc: 0.5528 - main_output_recall_m: 0.2153 - aux_output_acc: 0.5528 - aux_output_recall_m: 0.2183 - val_loss: 0.7422 - val_main_output_loss: 0.6730 - val_aux_output_loss: 0.6923 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 48/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7563 - main_output_loss: 0.6887 - aux_output_loss: 0.6765 - main_output_acc: 0.5000 - main_output_recall_m: 0.1667 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.2778WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7631 - main_output_loss: 0.6936 - aux_output_loss: 0.6954 - main_output_acc: 0.4907 - main_output_recall_m: 0.1532 - aux_output_acc: 0.4658 - aux_output_recall_m: 0.2013 - val_loss: 0.7416 - val_main_output_loss: 0.6721 - val_aux_output_loss: 0.6954 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.2500\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 49/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7679 - main_output_loss: 0.6970 - aux_output_loss: 0.7089 - main_output_acc: 0.4688 - main_output_recall_m: 0.2143 - aux_output_acc: 0.4688 - aux_output_recall_m: 0.4286WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7592 - main_output_loss: 0.6895 - aux_output_loss: 0.6970 - main_output_acc: 0.4720 - main_output_recall_m: 0.3385 - aux_output_acc: 0.4845 - aux_output_recall_m: 0.4583 - val_loss: 0.7405 - val_main_output_loss: 0.6708 - val_aux_output_loss: 0.6975 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.2500\n",
      "Epoch 50/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7675 - main_output_loss: 0.6973 - aux_output_loss: 0.7024 - main_output_acc: 0.4688 - main_output_recall_m: 0.2353 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.3529WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7553 - main_output_loss: 0.6860 - aux_output_loss: 0.6923 - main_output_acc: 0.5093 - main_output_recall_m: 0.2483 - aux_output_acc: 0.5155 - aux_output_recall_m: 0.3719 - val_loss: 0.7371 - val_main_output_loss: 0.6672 - val_aux_output_loss: 0.6982 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.5000\n",
      "Epoch 51/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7682 - main_output_loss: 0.6969 - aux_output_loss: 0.7128 - main_output_acc: 0.4688 - main_output_recall_m: 0.2941 - aux_output_acc: 0.3750 - aux_output_recall_m: 0.1765WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7619 - main_output_loss: 0.6928 - aux_output_loss: 0.6906 - main_output_acc: 0.4783 - main_output_recall_m: 0.2551 - aux_output_acc: 0.5093 - aux_output_recall_m: 0.4973 - val_loss: 0.7355 - val_main_output_loss: 0.6655 - val_aux_output_loss: 0.6994 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 52/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7868 - main_output_loss: 0.7175 - aux_output_loss: 0.6922 - main_output_acc: 0.4062 - main_output_recall_m: 0.2857 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.4286WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 14ms/step - loss: 0.7551 - main_output_loss: 0.6868 - aux_output_loss: 0.6825 - main_output_acc: 0.4534 - main_output_recall_m: 0.2651 - aux_output_acc: 0.5466 - aux_output_recall_m: 0.4587 - val_loss: 0.7379 - val_main_output_loss: 0.6677 - val_aux_output_loss: 0.7027 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.5556 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 53/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7360 - main_output_loss: 0.6688 - aux_output_loss: 0.6724 - main_output_acc: 0.5312 - main_output_recall_m: 0.5833 - aux_output_acc: 0.6875 - aux_output_recall_m: 0.7917WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 14ms/step - loss: 0.7569 - main_output_loss: 0.6891 - aux_output_loss: 0.6781 - main_output_acc: 0.4783 - main_output_recall_m: 0.4630 - aux_output_acc: 0.6087 - aux_output_recall_m: 0.5988 - val_loss: 0.7391 - val_main_output_loss: 0.6685 - val_aux_output_loss: 0.7061 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 54/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7682 - main_output_loss: 0.6987 - aux_output_loss: 0.6949 - main_output_acc: 0.4375 - main_output_recall_m: 0.4737 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.6316WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7637 - main_output_loss: 0.6953 - aux_output_loss: 0.6840 - main_output_acc: 0.4286 - main_output_recall_m: 0.3423 - aux_output_acc: 0.5342 - aux_output_recall_m: 0.7658 - val_loss: 0.7376 - val_main_output_loss: 0.6667 - val_aux_output_loss: 0.7089 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 55/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7675 - main_output_loss: 0.6977 - aux_output_loss: 0.6973 - main_output_acc: 0.5625 - main_output_recall_m: 0.4286 - aux_output_acc: 0.4062 - aux_output_recall_m: 0.5714WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7578 - main_output_loss: 0.6890 - aux_output_loss: 0.6883 - main_output_acc: 0.5217 - main_output_recall_m: 0.5293 - aux_output_acc: 0.5155 - aux_output_recall_m: 0.7668 - val_loss: 0.7378 - val_main_output_loss: 0.6666 - val_aux_output_loss: 0.7123 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 56/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7562 - main_output_loss: 0.6873 - aux_output_loss: 0.6881 - main_output_acc: 0.5312 - main_output_recall_m: 0.5263 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.5789WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7589 - main_output_loss: 0.6911 - aux_output_loss: 0.6785 - main_output_acc: 0.5404 - main_output_recall_m: 0.4126 - aux_output_acc: 0.5714 - aux_output_recall_m: 0.6605 - val_loss: 0.7388 - val_main_output_loss: 0.6672 - val_aux_output_loss: 0.7155 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.7500 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.7500\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 57/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7836 - main_output_loss: 0.7155 - aux_output_loss: 0.6814 - main_output_acc: 0.4375 - main_output_recall_m: 0.3529 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.7647WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7702 - main_output_loss: 0.7015 - aux_output_loss: 0.6861 - main_output_acc: 0.4596 - main_output_recall_m: 0.2840 - aux_output_acc: 0.5280 - aux_output_recall_m: 0.8573 - val_loss: 0.7419 - val_main_output_loss: 0.6701 - val_aux_output_loss: 0.7180 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 58/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7996 - main_output_loss: 0.7292 - aux_output_loss: 0.7037 - main_output_acc: 0.3438 - main_output_recall_m: 0.2632 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.7895WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7647 - main_output_loss: 0.6960 - aux_output_loss: 0.6870 - main_output_acc: 0.4472 - main_output_recall_m: 0.2249 - aux_output_acc: 0.5590 - aux_output_recall_m: 0.8550 - val_loss: 0.7436 - val_main_output_loss: 0.6717 - val_aux_output_loss: 0.7195 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 59/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7439 - main_output_loss: 0.6763 - aux_output_loss: 0.6758 - main_output_acc: 0.4688 - main_output_recall_m: 0.3810 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.8095WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7452 - main_output_loss: 0.6768 - aux_output_loss: 0.6841 - main_output_acc: 0.5342 - main_output_recall_m: 0.3041 - aux_output_acc: 0.5093 - aux_output_recall_m: 0.6903 - val_loss: 0.7437 - val_main_output_loss: 0.6716 - val_aux_output_loss: 0.7205 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 60/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7521 - main_output_loss: 0.6829 - aux_output_loss: 0.6918 - main_output_acc: 0.5938 - main_output_recall_m: 0.2667 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.9333WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7558 - main_output_loss: 0.6871 - aux_output_loss: 0.6870 - main_output_acc: 0.4783 - main_output_recall_m: 0.2426 - aux_output_acc: 0.5217 - aux_output_recall_m: 0.8906 - val_loss: 0.7445 - val_main_output_loss: 0.6726 - val_aux_output_loss: 0.7190 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 61/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7682 - main_output_loss: 0.6984 - aux_output_loss: 0.6978 - main_output_acc: 0.5000 - main_output_recall_m: 0.2667 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.8667WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7536 - main_output_loss: 0.6851 - aux_output_loss: 0.6849 - main_output_acc: 0.5093 - main_output_recall_m: 0.2640 - aux_output_acc: 0.5652 - aux_output_recall_m: 0.9214 - val_loss: 0.7452 - val_main_output_loss: 0.6733 - val_aux_output_loss: 0.7195 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 62/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7918 - main_output_loss: 0.7224 - aux_output_loss: 0.6937 - main_output_acc: 0.3125 - main_output_recall_m: 0.1053 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.8421WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7626 - main_output_loss: 0.6945 - aux_output_loss: 0.6810 - main_output_acc: 0.4596 - main_output_recall_m: 0.2304 - aux_output_acc: 0.5590 - aux_output_recall_m: 0.7358 - val_loss: 0.7458 - val_main_output_loss: 0.6737 - val_aux_output_loss: 0.7210 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 63/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7843 - main_output_loss: 0.7152 - aux_output_loss: 0.6913 - main_output_acc: 0.4062 - main_output_recall_m: 0.2222 - aux_output_acc: 0.5938 - aux_output_recall_m: 0.8889WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7625 - main_output_loss: 0.6936 - aux_output_loss: 0.6896 - main_output_acc: 0.5031 - main_output_recall_m: 0.2642 - aux_output_acc: 0.5404 - aux_output_recall_m: 0.9038 - val_loss: 0.7456 - val_main_output_loss: 0.6736 - val_aux_output_loss: 0.7199 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 64/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7516 - main_output_loss: 0.6829 - aux_output_loss: 0.6877 - main_output_acc: 0.5312 - main_output_recall_m: 0.2353 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.8235WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7572 - main_output_loss: 0.6884 - aux_output_loss: 0.6880 - main_output_acc: 0.4907 - main_output_recall_m: 0.2124 - aux_output_acc: 0.5466 - aux_output_recall_m: 0.7261 - val_loss: 0.7464 - val_main_output_loss: 0.6743 - val_aux_output_loss: 0.7211 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 65/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7382 - main_output_loss: 0.6704 - aux_output_loss: 0.6779 - main_output_acc: 0.5938 - main_output_recall_m: 0.3529 - aux_output_acc: 0.5938 - aux_output_recall_m: 0.8824WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7564 - main_output_loss: 0.6886 - aux_output_loss: 0.6777 - main_output_acc: 0.5031 - main_output_recall_m: 0.2212 - aux_output_acc: 0.5963 - aux_output_recall_m: 0.9248 - val_loss: 0.7483 - val_main_output_loss: 0.6760 - val_aux_output_loss: 0.7231 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 66/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7653 - main_output_loss: 0.6963 - aux_output_loss: 0.6901 - main_output_acc: 0.5938 - main_output_recall_m: 0.2857 - aux_output_acc: 0.4688 - aux_output_recall_m: 1.0000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7505 - main_output_loss: 0.6824 - aux_output_loss: 0.6809 - main_output_acc: 0.5280 - main_output_recall_m: 0.2515 - aux_output_acc: 0.5590 - aux_output_recall_m: 0.9539 - val_loss: 0.7486 - val_main_output_loss: 0.6761 - val_aux_output_loss: 0.7249 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n",
      "Epoch 67/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7721 - main_output_loss: 0.7058 - aux_output_loss: 0.6628 - main_output_acc: 0.3438 - main_output_recall_m: 0.2000 - aux_output_acc: 0.5938 - aux_output_recall_m: 0.9000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7599 - main_output_loss: 0.6908 - aux_output_loss: 0.6910 - main_output_acc: 0.4969 - main_output_recall_m: 0.2171 - aux_output_acc: 0.5093 - aux_output_recall_m: 0.9104 - val_loss: 0.7486 - val_main_output_loss: 0.6758 - val_aux_output_loss: 0.7279 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n",
      "Epoch 68/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7357 - main_output_loss: 0.6643 - aux_output_loss: 0.7142 - main_output_acc: 0.8125 - main_output_recall_m: 0.5714 - aux_output_acc: 0.3750 - aux_output_recall_m: 0.8571WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7548 - main_output_loss: 0.6857 - aux_output_loss: 0.6912 - main_output_acc: 0.5528 - main_output_recall_m: 0.2841 - aux_output_acc: 0.5093 - aux_output_recall_m: 0.7415 - val_loss: 0.7476 - val_main_output_loss: 0.6746 - val_aux_output_loss: 0.7307 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n",
      "Epoch 69/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7538 - main_output_loss: 0.6867 - aux_output_loss: 0.6709 - main_output_acc: 0.5938 - main_output_recall_m: 0.2857 - aux_output_acc: 0.4688 - aux_output_recall_m: 1.0000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7555 - main_output_loss: 0.6870 - aux_output_loss: 0.6847 - main_output_acc: 0.5280 - main_output_recall_m: 0.2540 - aux_output_acc: 0.5217 - aux_output_recall_m: 0.9444 - val_loss: 0.7468 - val_main_output_loss: 0.6738 - val_aux_output_loss: 0.7297 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n",
      "Epoch 70/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7501 - main_output_loss: 0.6820 - aux_output_loss: 0.6809 - main_output_acc: 0.5625 - main_output_recall_m: 0.2667 - aux_output_acc: 0.5000 - aux_output_recall_m: 1.0000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7596 - main_output_loss: 0.6911 - aux_output_loss: 0.6850 - main_output_acc: 0.5280 - main_output_recall_m: 0.2415 - aux_output_acc: 0.5652 - aux_output_recall_m: 0.8153 - val_loss: 0.7467 - val_main_output_loss: 0.6737 - val_aux_output_loss: 0.7301 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n",
      "Epoch 71/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7507 - main_output_loss: 0.6829 - aux_output_loss: 0.6781 - main_output_acc: 0.5312 - main_output_recall_m: 0.1765 - aux_output_acc: 0.6250 - aux_output_recall_m: 1.0000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7540 - main_output_loss: 0.6855 - aux_output_loss: 0.6846 - main_output_acc: 0.4907 - main_output_recall_m: 0.3722 - aux_output_acc: 0.5528 - aux_output_recall_m: 0.9306 - val_loss: 0.7463 - val_main_output_loss: 0.6734 - val_aux_output_loss: 0.7294 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n",
      "Epoch 72/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7626 - main_output_loss: 0.6944 - aux_output_loss: 0.6822 - main_output_acc: 0.5000 - main_output_recall_m: 0.3889 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.8333WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7497 - main_output_loss: 0.6810 - aux_output_loss: 0.6867 - main_output_acc: 0.5031 - main_output_recall_m: 0.2969 - aux_output_acc: 0.5217 - aux_output_recall_m: 0.7444 - val_loss: 0.7441 - val_main_output_loss: 0.6712 - val_aux_output_loss: 0.7294 - val_main_output_acc: 0.5556 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 73/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7273 - main_output_loss: 0.6611 - aux_output_loss: 0.6613 - main_output_acc: 0.6250 - main_output_recall_m: 0.4211 - aux_output_acc: 0.6250 - aux_output_recall_m: 1.0000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7431 - main_output_loss: 0.6749 - aux_output_loss: 0.6823 - main_output_acc: 0.5714 - main_output_recall_m: 0.3508 - aux_output_acc: 0.5714 - aux_output_recall_m: 0.9256 - val_loss: 0.7434 - val_main_output_loss: 0.6707 - val_aux_output_loss: 0.7271 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 1.0000\n",
      "Epoch 74/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7604 - main_output_loss: 0.6894 - aux_output_loss: 0.7100 - main_output_acc: 0.5938 - main_output_recall_m: 0.3077 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.9231WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7529 - main_output_loss: 0.6846 - aux_output_loss: 0.6823 - main_output_acc: 0.4969 - main_output_recall_m: 0.2305 - aux_output_acc: 0.5714 - aux_output_recall_m: 0.8028 - val_loss: 0.7451 - val_main_output_loss: 0.6723 - val_aux_output_loss: 0.7276 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 75/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7706 - main_output_loss: 0.7012 - aux_output_loss: 0.6945 - main_output_acc: 0.5938 - main_output_recall_m: 0.4000 - aux_output_acc: 0.5312 - aux_output_recall_m: 1.0000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7479 - main_output_loss: 0.6803 - aux_output_loss: 0.6758 - main_output_acc: 0.5528 - main_output_recall_m: 0.2645 - aux_output_acc: 0.5776 - aux_output_recall_m: 0.7666 - val_loss: 0.7453 - val_main_output_loss: 0.6726 - val_aux_output_loss: 0.7268 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 76/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7576 - main_output_loss: 0.6892 - aux_output_loss: 0.6840 - main_output_acc: 0.4062 - main_output_recall_m: 0.2941 - aux_output_acc: 0.5000 - aux_output_recall_m: 0.8235WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7509 - main_output_loss: 0.6822 - aux_output_loss: 0.6866 - main_output_acc: 0.5031 - main_output_recall_m: 0.2255 - aux_output_acc: 0.5466 - aux_output_recall_m: 0.8333 - val_loss: 0.7445 - val_main_output_loss: 0.6719 - val_aux_output_loss: 0.7261 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 77/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7587 - main_output_loss: 0.6912 - aux_output_loss: 0.6751 - main_output_acc: 0.4375 - main_output_recall_m: 0.1176 - aux_output_acc: 0.5938 - aux_output_recall_m: 0.8235WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 12ms/step - loss: 0.7511 - main_output_loss: 0.6828 - aux_output_loss: 0.6830 - main_output_acc: 0.4969 - main_output_recall_m: 0.2226 - aux_output_acc: 0.5217 - aux_output_recall_m: 0.6341 - val_loss: 0.7437 - val_main_output_loss: 0.6711 - val_aux_output_loss: 0.7262 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 78/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7732 - main_output_loss: 0.7060 - aux_output_loss: 0.6718 - main_output_acc: 0.3125 - main_output_recall_m: 0.2000 - aux_output_acc: 0.5938 - aux_output_recall_m: 0.7200WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7599 - main_output_loss: 0.6916 - aux_output_loss: 0.6834 - main_output_acc: 0.5466 - main_output_recall_m: 0.3061 - aux_output_acc: 0.5652 - aux_output_recall_m: 0.6633 - val_loss: 0.7434 - val_main_output_loss: 0.6709 - val_aux_output_loss: 0.7255 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 79/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7424 - main_output_loss: 0.6748 - aux_output_loss: 0.6765 - main_output_acc: 0.4688 - main_output_recall_m: 0.3333 - aux_output_acc: 0.6250 - aux_output_recall_m: 0.8571WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7465 - main_output_loss: 0.6783 - aux_output_loss: 0.6815 - main_output_acc: 0.5590 - main_output_recall_m: 0.3593 - aux_output_acc: 0.5901 - aux_output_recall_m: 0.6586 - val_loss: 0.7432 - val_main_output_loss: 0.6708 - val_aux_output_loss: 0.7238 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.4444 - val_aux_output_recall_m: 0.7500\n",
      "Epoch 80/80\n",
      "1/6 [====>.........................] - ETA: 0s - loss: 0.7609 - main_output_loss: 0.6923 - aux_output_loss: 0.6857 - main_output_acc: 0.5000 - main_output_recall_m: 0.4000 - aux_output_acc: 0.5312 - aux_output_recall_m: 0.6000WARNING:tensorflow:Early stopping conditioned on metric `val_prc` which is not available. Available metrics are: loss,main_output_loss,aux_output_loss,main_output_acc,main_output_recall_m,aux_output_acc,aux_output_recall_m,val_loss,val_main_output_loss,val_aux_output_loss,val_main_output_acc,val_main_output_recall_m,val_aux_output_acc,val_aux_output_recall_m\n",
      "6/6 [==============================] - 0s 13ms/step - loss: 0.7497 - main_output_loss: 0.6811 - aux_output_loss: 0.6863 - main_output_acc: 0.5714 - main_output_recall_m: 0.3866 - aux_output_acc: 0.5466 - aux_output_recall_m: 0.5538 - val_loss: 0.7427 - val_main_output_loss: 0.6707 - val_aux_output_loss: 0.7208 - val_main_output_acc: 0.6667 - val_main_output_recall_m: 0.5000 - val_aux_output_acc: 0.3333 - val_aux_output_recall_m: 0.5000\n"
     ]
    }
   ],
   "source": [
    "\n",
    "# reduce_lr = ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)\n",
    "# checkpointer = ModelCheckpoint(filepath='lstm.hdf5', verbose=1, save_best_only=True)\n",
    "import tensorflow as tf\n",
    "\n",
    "early_stopping = tf.keras.callbacks.EarlyStopping(\n",
    "    monitor='val_prc', \n",
    "    verbose=1,\n",
    "    patience=10,\n",
    "    mode='max',\n",
    "    restore_best_weights=True)\n",
    "\n",
    "history=model.fit({'main_input': dframe_train_X_time.astype(np.float32), 'aux_input': dframe_train_static.astype(np.float32)},\n",
    "          {'main_output':dframe_train_Y.astype(np.float32), 'aux_output': dframe_train_Y.astype(np.float32)},\n",
    "          epochs=80 ,callbacks=[early_stopping],validation_split=0.05)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "5a7cf7fe",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYIAAAEWCAYAAABrDZDcAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA8g0lEQVR4nO3dd3wc1bn/8c+z0qr3ZlX3XuWOC2C6Cx1MJwkXQkgIIbmEALnp5V7yCykQWkhCEgIxndAMGAym2QbLXe5VVrElWb1Lu3t+f8zKVrUlW6vd1T7v12tfkmZmZ5+V7PnuOWfmjBhjUEopFbhs3i5AKaWUd2kQKKVUgNMgUEqpAKdBoJRSAU6DQCmlApwGgVJKBTgNAjUgiciTIvJjb9fRkYj8TESe7eG2q0TkttPdj1InE+ztApTyBGPMHd6uQSl/oS0CpZQKcBoEymeIyEERuVdEtohInYj8TUQGicg7IlIjIh+ISHyb7V8SkSMiUiUin4jIhDbr/iEiv3J/v0BECkTkHhEpEZHDInLLCepYJSK/EpHVIlIrIm+KSKKIPCci1SKyTkSGttl+rntZlfvr3DbrhonIx+763weSOrzWGe7XqRSRzSKy4BR/d5eKyDb3flaJyLg26+4TkUJ3DbtE5Dz38lkikuN+T8Ui8vtTeW3l/zQIlK+5CrgAGA1cArwD/BDrAGoDvtNm23eAUUAKsAF47gT7TQVigQzgVuCxtqHSheuAm93bjwDWAH8HEoAdwE8BRCQBeBt4BEgEfg+8LSKJ7v38G1jvrv+XwFdbX0BEMtzP/ZV7v98HXhGR5BPU1YmIjAaWAd8FkoHlwJsiEiIiY4BvAzONMdHARcBB91MfBh42xsS43+OLvXldNXBoEChf8ydjTLExphD4FPjCGLPRGNMEvAZMbd3QGPO0MabGve5nwBQRie1mvy3AL4wxLcaY5UAtMOYEdfzdGLPPGFOFFTj7jDEfGGMcwEtt6lgC7DHG/MsY4zDGLAN2ApeIyGBgJvBjY0yTMeYT4M02r3ETsNwYs9wY4zLGvA/kAIt7/NuyXAu8bYx53xjTAjwEhANzAScQCowXEbsx5qAxZl+b38lIEUkyxtQaY9b28nXVAKFBoHxNcZvvG7r4OQpARIJE5EER2Sci1Rz/lNuu66WNMvdBvFV9675Opw4gHcjr8Nw8rJZEOlBhjKnrsK7VEGCpuzunUkQqgflA2gnq6kq7GowxLiAfyDDG7MVqKfwMKBGR50Uk3b3prVgtr53uLq2Le/m6aoDQIFD+6gbgMuB8rC6foe7l0s91FGEd0NsaDBQCh4F4EYnssK5VPvAvY0xcm0ekMebB06lBRATIcteAMebfxpj57m0M8Bv38j3GmOuxutZ+A7zcoVYVIDQIlL+KBpqAMiAC+F8v1bEcGC0iN4hIsIhcC4wH3jLG5GF19fzc3V8/H2vco9WzWF1IF7lbOGHuge3MXtbwIrBERM4TETtwD9bvZrWIjBGRc0UkFGjEas04AUTkJhFJdrcgKt37cp7Sb0H5NQ0C5a+eweoOKQS2A17p3zbGlAEXYx18y4AfABcbY466N7kBmA2UYw0wP9PmuflYrZofAqVYLYR76eX/S2PMLqzxhj8BR7HC5hJjTDPW+MCD7uVHsD79/9D91IXANhGpxRo4vs4Y09ib11YDg+iNaZRSKrBpi0AppQKcBoFSSgU4DQKllApwGgRKKRXg/G720aSkJDN06FBvl6GUUn5l/fr1R40xXU5f4ndBMHToUHJycrxdhlJK+RUR6XgF/DHaNaSUUgFOg0AppQKcBoFSSgU4vxsj6EpLSwsFBQU0Ng78q+PDwsLIzMzEbrd7uxSl1AAxIIKgoKCA6Ohohg4dijXx4sBkjKGsrIyCggKGDRvm7XKUUgPEgOgaamxsJDExcUCHAICIkJiYGBAtH6VU/xkQQQAM+BBoFSjvUynVfwZMEJxUSwNUFYDL5e1KlFLKpwROEDiboa4Ummv7fNeVlZU8/vjjvX7e4sWLqays7PN6lFKqNwInCEKiAIGm6j7fdXdB4HSe+GZPy5cvJy4urs/rUUqp3hgQZw31iC0IQqOgsdq6w20fuv/++9m3bx/Z2dnY7XaioqJIS0tj06ZNbN++ncsvv5z8/HwaGxu5++67uf3224Hj02XU1tayaNEi5s+fz+rVq8nIyOD1118nPDy8bwtVSqkuDLgg+Pmb29he1M2nfmcLOJvAXgHS88bQ+PQYfnrJhG7XP/jgg+Tm5rJp0yZWrVrFkiVLyM3NPXaK59NPP01CQgINDQ3MnDmTq666isTExHb72LNnD8uWLeMvf/kL11xzDa+88go33XRTj2tUSqlTNeCC4IRsQdatuY2zV0HQW7NmzWp3nv8jjzzCa6+9BkB+fj579uzpFATDhg0jOzsbgOnTp3Pw4EGP1aeUUm0NuCA40Sd3AIq3QXAYJI7wWA2RkZHHvl+1ahUffPABa9asISIiggULFnR5HUBoaOix74OCgmhoaPBYfUop1VbgDBa3CouBpto+PY00OjqampqaLtdVVVURHx9PREQEO3fuZO3atX32ukop1RcGXIvgpEJjoe6odRppWEyf7DIxMZF58+YxceJEwsPDGTRo0LF1Cxcu5Mknn2Ty5MmMGTOGM844o09eUyml+ooYY7xdQ6/MmDHDdLwxzY4dOxg3blzPduBywZEtEJkEsZkeqNDzevV+lVIKEJH1xpgZXa0LvK4hm+34aaRKKaUCMAjA6h5yNoGjyduVKKWU1wVMEDhdLmoaW3AZA2HR1kJtFSilVOAEQXWjgwNH62hqcVmnjwaFemS6CaWU8jcBEwTh9iAAGlrc8/+ExUBTDbhOPB+QUkoNdAETBKHBNmwix4MgNAYw1jUFSikVwAImCESEcHsQDc2tQRAFEgRNlae971Odhhrgj3/8I/X19addg1JKnSqPBYGIPC0iJSKS2836G0Vki/uxWkSmeKqWVuEhQTS2ODHGWHMNhcZYA8aneS2FBoFSyp958srifwCPAs90s/4AcLYxpkJEFgFPAbM9WA/h9iBcxtDkcBFmD7LGCRoroLnOaiGcorbTUF9wwQWkpKTw4osv0tTUxBVXXMHPf/5z6urquOaaaygoKMDpdPLjH/+Y4uJiioqKOOecc0hKSuKjjz7qw3erlFI947EgMMZ8IiJDT7B+dZsf1wJ9c5nvO/fDka1droo1BnuzkyC7zbqwDGOFQJDdOouoO6mTYNGD3a5uOw31ihUrePnll/nyyy8xxnDppZfyySefUFpaSnp6Om+//TZgzUEUGxvL73//ez766COSkpJO510rpdQp85UxgluBd7pbKSK3i0iOiOSUlpae8ouIWA+Xq7UrSKxxApfjlPfZ0YoVK1ixYgVTp05l2rRp7Ny5kz179jBp0iQ++OAD7rvvPj799FNiY/v47jhKKXWKvD7pnIicgxUE87vbxhjzFFbXETNmzDhxh/4JPrkLcLikFgFGpLi7gupKrZvaJ48De1gvq++yVh544AG+8Y1vdFq3fv16li9fzgMPPMCFF17IT37yk9N+PaWUOl1ebRGIyGTgr8Blxpiy/njN8JAgGloHjMGabgKgseqU99l2GuqLLrqIp59+mtpa67TUwsJCSkpKKCoqIiIigptuuonvf//7bNiwodNzlVLKG7zWIhCRwcCrwM3GmN399brh9iDK2g4YB4dAcLgVBNGDTr6DLrSdhnrRokXccMMNzJkzB4CoqCieffZZ9u7dy7333ovNZsNut/PEE08AcPvtt7No0SLS0tJ0sFgp5RUem4ZaRJYBC4AkoBj4KWAHMMY8KSJ/Ba4C8txPcXQ3RWpbpzsNdUOzkz0lNQxOiCAuIsRaWHMYao7AoInWwLGP02molVK9daJpqD151tD1J1l/G3Cbp16/O2F26wrj+mYncRHuhaGxVhA0Vln3KVBKqQDiK2cN9RsRIcwedHyqCQB7OASFnNY4gVJK+asBEwS96eIKt9tobG4zYCwCYbF+MQmdv91RTinl+wZEEISFhVFWVtbjg2R4SBBOY2h2tLmBvR9MQmeMoaysjLCw0z/NVSmlWnn9OoK+kJmZSUFBAT292KzZ4aKkpomWshAiQqzpqTEGqo/C4XqISPBgtacnLCyMzEz/vNeyUso3DYggsNvtDBs2rMfbNztcLP3pe9wybygPLG5z9s1Lv4WDn8E9u9xTUCil1MAXkEe7kGAbY9Oi2VrYYXB4zGKoK4GiDd4pTCmlvCAggwBgQnosuYVVOJxtxglGnW/NPbRrufcKU0qpfhawQXDu2BSqGx28sqHg+MLweBgyF3Z1O/+dUkoNOAEbBOePSyE7K44/vL+HxrbXFIxZDCXbofyA94pTSql+FLBBICLct3AsR6ob+efqg8dXjFlofd39rlfqUkqp/hawQQAwZ0QiZ49O5vFV+6hqaLEWJgyH5LE6TqCUChgBHQQAP1g4hqqGFv788b7jC8csgoOfQ0OF9wpTSql+EvBBMCE9lsuy03n68wMUVzdaC8csBuOEvSu9W5xSSvWDgA8CgHsuGIPTZXh45R5rQcZ0iEyGnW97tzCllOoHGgTA4MQILp6cznu5R6wFtiAYe7E1YNxc593ilFLKwzQI3EamRFFW10x9s/tG9pOWQku9XlOglBrwNAjcshKsu9QUVjRYCwbPgZgM2PqSF6tSSinP0yBwy4wPB6CgNQhsNph4Jez9AOrLvViZUkp5lgaB2/EgqD++cNJScDlg++teqkoppTxPg8AtOSqU0GAb+a0tAoDUyZA0Gra+7L3ClFLKwzQI3ESEjPjw9i0CEatVkPc5VBV0/2SllPJjGgRtZMZHHB8jaDXxKsBA7qteqUkppTxNg6CNzPjwzkGQOALSp+nZQ0qpAUuDoI3M+HDK65qpa3K0XzFpKRzZAqW7vVOYUkp5kAZBG5nx7msJKjt2D10JiLYKlFIDkseCQESeFpESEcntZv1YEVkjIk0i8n1P1dEbXZ5CChCdCsMXwObnweXq/ESllPJjnmwR/ANYeIL15cB3gIc8WEOvZLlbBPnlDZ1XZt8IVYesM4iUUmoA8VgQGGM+wTrYd7e+xBizDmjxVA29lRQVQmiwrXOLALg3N5OW4CjY9G8vVKaUUp7jF2MEInK7iOSISE5paaknX6fLM4dKa5p4aUs5OVELrKuMm2o9VoNSSvU3vwgCY8xTxpgZxpgZycnJHn2trq4l2HDIulPZ+yHnQUudTjmhlBpQ/CII+lNmx6uLgQ15VhB80TLSuqexdg8ppQYQDYIOMuMjqKhvobbNtQTr3UFQVtcC2TdA3mdQcdBLFSqlVN/y5Omjy4A1wBgRKRCRW0XkDhG5w70+VUQKgP8GfuTeJsZT9fRU6ymkrfclaHI42VJYhQiU1TVhJl8LiHUqqVJKDQDBntqxMeb6k6w/AmR66vVPVdtrCcakRrOtqJpmh4vpQ+JZn1dBTVgaMcPOsrqHzvqBdd8CpZTyY3oU6yDz2LUE1jhB6/jABeMHAVBW22xdU1CZp9cUKKUGBA2CDpKiQgiz246dObThUAWZ8eGMS7N6rcpqm2DcJRAaCzlPe7NUpZTqExoEHVjXElinkBpjWJ9XwfQh8SRGhgBQVtcMIREw7WbrNNKqQi9XrJRSp0eDoAuZ8eEUVNZTWNlAcXWTFQRR7iCobbY2mvV1wEDO37xXqFJK9QENgi60Xl3cetrotMHxJLS2CGqbrI3ih8KYxZDzd2jpYm4ipZTyExoEXciMj6CyvoWPd5cSERLE2NRoQoODiA4NtrqGWs3+BjSU6z2NlVJ+TYOgC62nkK7YVkx2VhzBQdavKTEqpH0QDD0TUibAF0+CMd4oVSmlTpsGQRdaTyGtbXIwfUj8seWJUaHHu4bAurn97G9Aca6eSqqU8lsaBF1obREATGsTBAmRIZS3bREATL4GwuNh7RP9VZ5SSvUpDYIuJEZa1xIATMs6HgRJUSEcre0QBPZwmP412LUcKvL6sUqllOobGgRdaL2WYGRKFLER9mPLEyNDqahvxuXqMB4w8+sgQbD6T/1cqVJKnT6PzTXk7753/mjsQdJuWUJkCE6XoaqhhXj36aQAxGbAlOtgwzNw1r0QPaifq1VKqVOnLYJuLJmcxoUTUtstO3ZRWV1T5yfM/x64WmDNo/1RnlJK9RkNgl5IigoF6DxOAJA4AiZcac0/VN/trZqVUsrnaBD0QuvVxZ3OHGp15j3QXAtf/Lkfq1JKqdOjQdALx+cb6qJrCGDQeBizxLrArKmmHytTSqlTp0HQCwkRVhB02TXU6qx7oLES1ulkdEop/6BB0AvBQTbiIuzddw0BZEyH4edYg8Y6GZ1Syg9oEPRSYmRI12cNtXXW96GuFDY91z9FKaXUadAg6CVrvqETtAgAhsyDzJnw+SPgdPRPYUopdYo0CHrJahGcJAhErOsKKvNg+3/6pS6llDpVGgS9lBgV0v1ZQ22NXgRJY+CzP+oU1Uopn6ZB0EuJkaFUNrTgcLpOvKHNBvO/C8VbYe/Kdqu2F1Xz+d6jnitSKaV6QYOglxKjQjAGKupbTr7xxKshJgM++0O7xT97cxv3v7rFQxUqpVTveCwIRORpESkRkdxu1ouIPCIie0Vki4hM81QtfSkx0ppm4qRnDgEEh8Ccb0PeZ5C/DoDGFiebDlVSXN2E0S4jpZQP8GSL4B/AwhOsXwSMcj9uB/zizi6tVxeXn+zMoVbTvmLduMbdKtiQV0Gz00Wzw0V1g55RpJTyPo8FgTHmE+BEs69dBjxjLGuBOBFJ81Q9fSXRPd/Q0Q5nDlU1dNNVFBoFM2+zblxTVcDa/WXHVpXUNHqsTqWU6ilvjhFkAPltfi5wL+tERG4XkRwRySktLe2X4rqT6J6BtO2ZQ1sKKpn6ixWs3tfNAHD2DYCBrS+zdn/5sfsclNT0oHtJKaU8zJtBIF0s67LT3BjzlDFmhjFmRnJysofLOrG4cDs2aT8D6eubinAZeHPz4a6flDAcMmfh2vIiG/MrOHt0CqAtAqWUb/BmEBQAWW1+zgSKvFRLj9lsQkLk8XsXG2N4N/cIAB/sKO58G8tWk6/BVrKNEa48LstOB6CkWlsESinv82YQvAF8xX320BlAlTGmm4/UviUxMvRY19CWgioKKxs4c1QSpTVNbC6o7PpJE67AKUFcEfw5C8YkE24P0q4hpZRP8OTpo8uANcAYESkQkVtF5A4RucO9yXJgP7AX+AvwLU/V0tcSo0KOdQ0tzz1MsE349eWTCLIJ728v7vpJkUlstE/nKvsaokOCSIkJ1SBQSvkEj9283hhz/UnWG+BOT72+JyVEhrCtqBpjDO9sPcK8kUkMToxg1tAE3t9ezA8Wju30nPpmB8/Wz+aPwV9C3uekRIdSqmMESikfoFcWn4KkqFCO1jaxraiaQ+X1LJ5k3eT+gvGD2FNSy8GjdZ2esz6vgncd03AER8KWF0iJDtMWgVLKJ/QoCETkbhGJcffn/01ENojIhZ4uzlclRoZQ0+jgjc1FBNmEC8YfDwKgy+6htfvLaLGFYcZdAttfJy0SSnWwWCnlA3raIvgvY0w1cCGQDNwCPOixqnxcgvvq4pdy8jljeMKxm9pnJUQwNjW6myAoZ3JmLPbsa6GpmunN66hpctDQ7OzX2pVSqqOeBkHrOf+Lgb8bYzbT9XUAAaF1vqGK+hYWTWx/MfSF4weRk1fe7jqDuiYHm/MrOWN4Igw7G6IGMbnsXUCvJVBKeV9Pg2C9iKzACoL3RCQaOMk8zANXkrtFIAIXTUhtt+7CCam4DKzccbxVsD6vAofLMGd4ItiCYOJVpJV+SjT1Ok6glPK6np41dCuQDew3xtSLSAJW91BAau0Kmjk0geTo0HbrJqTHkB4bxvvbi5mUGcuyLw7x6sZCwu1BTB8S797oCmxrH+c82wZKquf3d/lKKdVOT4NgDrDJGFMnIjcB04CHPVeWb0uNDSM6LJil0zM7rRMRzh8/iH+tzWPF9mJCgmwsmpTKrfOHERnq/nVnzMAZnc6SyrUU1Nzez9UrpVR7PQ2CJ4ApIjIF+AHwN+AZ4GxPFebLIkKCWf+jCwgJ7rpn7YbZg9ldXMP54wZx1bRM4t0tiGNsNmzjL+fstU/xWEUZMMzzRSulVDd6OkbgcF8AdhnwsDHmYSDac2X5vu5CAGBsagzP3z6H284c3jkE3GTiFYSIg0FFH3qqRKWU6pGeBkGNiDwA3Ay8LSJBgN1zZQWAjBkctSUxtmLlybdVSikP6mkQXAs0YV1PcATrvgG/9VhVgcBmY0PU2UxsyIHGKm9Xo5QKYD0KAvfB/zkgVkQuBhqNMc94tLIAsC/lAkJwwK53vF2KUiqA9XSKiWuAL4GlwDXAFyJytScLCwSNKVMpNIm4cl/1dilKqQDW07OG/geYaYwpARCRZOAD4GVPFRYIUmLDecc5i1v3r4SGSgiP83ZJSqkA1NMxAltrCLiV9eK5qhsp0WG87TwDcTZr95BSymt6ejB/V0TeE5GvicjXgLexbiyjTkNKdCgbzUgaItIh95Vjy50ug3W2rlJKeV5PB4vvBZ4CJgNTgKeMMfd5srBAkBITCgj70hbDvpVQXYTD6WLBQx/x5Mf7vV2eUipA9Lh7xxjzijHmv40x3zPGvObJogJFUlQoIvBl7BIwLtj0HJvyK8kvb+DDnd3c8lIppfrYCQeLRaQG6KqPQrDuNhnjkaoChD3IRkJECHscyTD0TNjwLz6sWwLA5oIqmhxOQoODvFylUmqgO2GLwBgTbYyJ6eIRrSHQN5KjQymtaYJpX4HKPMpyVxIabKPZ4SK3sNrb5SmlAoCe+eNlKTFh1k3sx12CKySGOdXL+ercoQCszyv3bnFKqYCgQeBlyVGh1s1p7OHsTV3EIts6rp0YxZDECHIOVni7PKVUANAg8LKUGKtryOUyPO9YQKi0MPzwO0wfEs/6vAo9jVQp5XEaBF6WEh2Kw2UormlkWX4Ch8NHIRv/xYwhCZTVNXOwrN7bJSqlBjiPBoGILBSRXSKyV0Tu72J9vIi8JiJbRORLEZnoyXp8UUp0GABvbi6iocVJzfjr4cgW5kUWApBzUMcJlFKe5bEgcN+z4DFgETAeuF5ExnfY7IdYt8CcDHyFALz9pXVRGTy/Lp8wu43BZ38NgsMYvP95YsKCWZ+n4wRKKc/yZItgFrDXGLPfGNMMPI91h7O2xgMrAYwxO4GhIjLIgzX5nJRoKwj2l9Yxd0QSYTGJMOU6ZPMyzsmEddoiUEp5mCeDIAPIb/NzgXtZW5uBKwFEZBYwBOh8R/gBrLVrCOCcsSnWN3PuAmczN9veY19pHRV1zV6qTikVCDwZBNLFso6nwDwIxIvIJuAuYCPg6LQjkdtFJEdEckpLS/u8UG8KDwkiOtS6wPvc1iBIGgljl5B95GUiaNTuIaWUR3kyCAqArDY/ZwJFbTcwxlQbY24xxmRjjREkAwc67sgY85QxZoYxZkZycrIHS/aO5JhQxgyKJiMu/PjCed8luLmKG+yryNEgUEp5kCeDYB0wSkSGiUgIcB3wRtsNRCTOvQ7gNuATY0zAzavww0Xj+OmlHcbRs2bC4Lncbn+HTQdLun6iUkr1AY8FgTHGAXwbeA/YAbxojNkmIneIyB3uzcYB20RkJ9bZRXd7qh5fdv74QcwdkdR5xbzvkOIqJaPoPZoczv4vTCkVEMTfrlydMWOGycnJ8XYZ/cPlovYPM8ivaqH+vz5m+tAEb1eklPJTIrLeGDOjq3V6ZbEvs9kwc7/DONsh9q5909vVKKUGKA0CHxc983rqbNHE7HqRZofL2+UopQYgDQJfFxxK1bDFnO1ax/ub9faVSqm+p0HgB1Ln3UyENLH30xe9XYpSagDSIPADtqHzqA1NYULZCnYX13i7HKXUAKNB4A9sNoImL+Vs2xZe/WyLt6tRSg0wGgR+InzaddjFSfOW16hr6jQLh1JKnTINAn+ROomG2JFcZD7ljc1FJ99eKaV6SIPAX4gQNu1aZtt28u7n6/QWlkqpPqNB4Edk0tUAjD36PtsPB9yUTEopD9Eg8CcJw2kcNJXLglbr2UNKqT6jQeBngqZcy3hbHnX5W71dilJqgNAg8DP2KdfQhJ0RB5Z5uxSl1AChQeBvIhP5LOwcpla+Cw16wxql1OnTIPBD69OuI8w0wYZnvF2KUmoA0CDwQ8Hpk1jjGo/58ilw6sVlSqnTo0Hgh7Liw/m74yKkqgB2vuXtcpRSfk6DwA9lJUTwgWs6DVFZ8MWT3i5HKeXnNAj80OCECFzY2JF5LRxaA0WbvF2SUsqPaRD4oUExYdiDhE+iFoI9UlsFSqnTokHgh4JsQkZcOHuqgyD7Bsh9BWqKvV2WUspPaRD4qayECArK62H2HeBshpy/ebskpZSf0iDwU1kJEeRXNEDSSBi9CNb9FVoavF2WUsoPaRD4qaz4CMrrmqltcsCcO6G+DLa84O2ylFJ+SIPAT2UlhAOQX14PQ+dD6iRY8zjofQqUUr3k0SAQkYUisktE9orI/V2sjxWRN0Vks4hsE5FbPFnPQDI4IQJwB4EIzPk2HN0Fe1d6uTKllL/xWBCISBDwGLAIGA9cLyLjO2x2J7DdGDMFWAD8TkRCPFXTQJIV7w6CCve4wIQrISoV1jzqxaqUUv7Iky2CWcBeY8x+Y0wz8DxwWYdtDBAtIgJEAeWATp7TA3ERdqJCg60WAUBwCMy+HfZ/BMXbvFucUsqveDIIMoD8Nj8XuJe19SgwDigCtgJ3G2NcHXckIreLSI6I5JSWlnqqXr8iItaZQ61BADD9FrBHwNrHvVeYUsrveDIIpItlHUcyLwI2AelANvCoiMR0epIxTxljZhhjZiQnJ/d1nX4rKz6c/Io2QRCRANk3wuYXtFWglOoxTwZBAZDV5udMrE/+bd0CvGose4EDwFgP1jSgWC2CBkzbM4UWPABhsfD6nTpFtVKqRzwZBOuAUSIyzD0AfB3wRodtDgHnAYjIIGAMsN+DNQ0oWfHhNLQ4OVrbfHxhZCIs/i0UbYS1j3mvOKWU3/BYEBhjHMC3gfeAHcCLxphtInKHiNzh3uyXwFwR2QqsBO4zxhz1VE0DzeDE1jOH6tuvmHAFjL0YPvw1HN3jhcqUUv4k2JM7N8YsB5Z3WPZkm++LgAs9WcNAduwU0vJ6pg2OP75CBJb8Dh6bBW/cBV9bDja9dlAp1TU9OvixzDZB0El0Kix80Lpfwbq/9HNlSil/okHgx8JDgkiODiW/vJvJ5qZcD6MuhBU/grzV/VucUspvaBD4uU6nkLYlAlf8GeKGwPM3wNG9/VucUsovaBD4uayECA511TXUKiIBbnwRxAb/Xgp1Zf1XnFLKL2gQ+Lms+AgOVzXy6Z5S9pXW0tDs7LxRwnC4bhlUFVotg5bG/i9UKeWzPHrWkPK8iRkxOF2Gm//25bFlI1Oi+N3SKUzJiju+4eDZcMWT8PIt8K/LYdFvIG1Kv9erlPI9Yvxs/voZM2aYnJwcb5fhUworGygor6eoqoGiykb+/cUhiqsbuW/hWG6dPwybrc1sH5uWwXs/hIYKyL6RA1O+R0r6UCJD9TOBUgOZiKw3xszocp0GwcBTWd/Mfa9s4b1txSwYk8zvlk4hMSr0+AYNlfDpQ7jW/plGp/Dh4Lu4+NYfe61epZTnnSgIdIxgAIqLCOHJm6bzy8smsHpfGVc9sZqKujbTUITH0XLeL7gt6nG+dI3l4vyHcL75PXC2eK9opZTXaBAMUCLCzXOG8txtsymqbOTOf2+gxXl8hu8/f7yPD0sieGfyH3nScQlB65+GZ6+E+nIvVq2U8gYNggFu5tAE/vfKSazeV8Yv39oOwO7iGh5ZuZeLJ6fxyyuyecL+FZ5LewAOrYW/ngcHPvVy1Uqp/qRBEACunp7J188cxjNr8nhmzUHufWkz0WHB/PzSCYQE21g8KY1fF2bTeOMb1qml/7wY/nExHPzM26UrpfqBBkGAuH/ROBaMSeYnr29jc0EVP79swrEB5Muz06lvdvJe9WD4zkZY+Btr1tJ/LLEC4fAWL1evlPIkDYIAEWQTHrl+KhMzYrhyagZLJqUdWzdzaALpsWH8Z2Mh2MPgjDvg7k1WIJTsgKfOhuU/YHdePnct28gzaw52PdGdUsov6cnjASQmzM4bd85HxBpMbmWzCZdmZ/CXT/dTVttktRTs4VYgTLnWuq/Bur+QtuElbPXX8ZPN84FtjEiO5CtzhvLVuUO99p6UUqdPWwQBxmaTdiHQ6vKp6Thdhre3Hm6/IjweljyE+fqH5LsSeTjkcTbP+oCfLhlDZGgwP3tzGyXVOmWFUv5Mg0ABMDY1hjGDoq3uoS4cDBnNxfU/ZfuQm4nd8jS35P8Pv798JMbAe9uO9HO1Sqm+pEGgjrlsajobDlVyqKxz//8nu0txYSPy0t/A4odgzwpGvrWU2UlNLN+qQaCUP9MgUMdclp2BCLy8oaDTuk92lzIkMYIhiZEw6+tw/QtQvp+nm+4hJe9NjtZo95BS/kqDQB2TERfOWaOSeWHdIRxtrkJudrhYs7+Ms0YlH9949IVw6wpscZk8bH8Uxz8uhdLdXqhaKXW6NAhUOzfMHkxxdRMf7iw5tiwnr5z6ZidnjU5uv/GgCYTd8RF/CL2D6PJceGIuvHOfdcqpUspvaBCods4bm8KgmFD+/eWhY8s+2X2UYJswZ0Rip+0lKJiWqbdwTtNDNI2/Ctb9FR4/A/5yLuQ8bd0es7EK/GyWW6UCiV5HoNoJDrJx7Yws/vTRXgoq6smMj+CT3aVMHxJPVDf3LFg8KY3HV8Xy+pAfcc2iX8OWF2Djs/DW99rsOAwiUyAqBaJTIWoQRKdB4nBIGg0JIyAkop/epVKqLQ0C1cm1swbz6Ed7eWFdPl+ZM5Tth6u596Ix3W4/IT2GrIRwluce5pqZs2DOnXDGt+DIFijZCXUlUFsMte6vZfsg73Pr5jjHCMQPhbFLYNLVkJYNXVzvoJTqex4NAhFZCDwMBAF/NcY82GH9vcCNbWoZByQbY3QuZC/KiAtnwZgUXliXT1aC9Sn97I7jA22ICIsnpvH05weoqm8hNsJuHcTTppz4dpgtDVYolO2x5jYqyIEv/gxrHrVaCNnXw5xvW1c5K6U8xmNjBCISBDwGLALGA9eLyPi22xhjfmuMyTbGZAMPAB9rCPiG62cNpqSmid+v2E1iZAjj02JOuP2iSWm0OA0f7Cjucn1lfTMf7ixmzb4yiqsbMcZYB/jUiTDhCjj7B3Dji/D93XDJIxCTDh/+yhqA1mmxlfIoT7YIZgF7jTH7AUTkeeAyYHs3218PLPNgPaoXzhmTTGpMGEeqG7k8O739fY+7MCUzlvTYMP7vnR28vfUwWfHhZMZHUFjZwNr9Zew8UtNu+4iQIEYkR3HvRWPan40UkQDTv2o99n8Mb37HmhZ7+tfggl9AWKwH3q06HcYY/vjBHr44UMZ/zRvG+eMGnfTfi/ItHrtnsYhcDSw0xtzm/vlmYLYx5ttdbBsBFAAju2oRiMjtwO0AgwcPnp6Xl+eRmlV7f3h/Nw+v3MPvr5nCldMyT7r9im1HeHl9AfkVDRSU11PT5CDMbmPGkARmD0tg5rAEmh0uDpbVceBoHR/vLuXg0Tp+uHgct84f1uUcSKa5jvK3fk78lr/QGBJPxHn3WaEQHNq5gFO04VAFK3cUc7SmmbK6Jo7WNpMWG8Z9C8cyNCmyz15nIHI4Xfzwta28mFNAXISdyvoWxqZG861zRrJkUhpBGgg+wys3rxeRpcBFHYJgljHmri62vRa4yRhzycn2qzev7z8Vdc08vmov3z1/NJHdnDHUHWMM1Q0OwkOCCAnuugeyrsnBPS9u5t1tR7hqWia/vmIiYfYgjtY2saWgkrX7y3k39wiHyuuZJPv5H/tznGHbAbFZVlfSlBsg6NQatcYY1uwv49EP97J6XxnBNiExKoSEyFASI0PYlF9Js9PFnQtGcseC4YQGB3W5nwNH6/jTyj0MSYxk6YxM0uP6bjzD6TL86u3tnDkqiXPHDuryPeSV1ZOVENGnB1xjDJ/uOcoLOfmcNSqJq6dndbn/hmYndy3bwAc7SvjOeaO469yRvLWliMc+2sfeklrOHp3MP26Z2WXAn0hJTSMv5RQQGmwjMSqExMjQ41e1q1PmrSCYA/zMGHOR++cHAIwx/9fFtq8BLxlj/n2y/WoQDCwul+GRD/fwxw/2MDwpkiaHi8LKBgDsQcLcEUksnJjK3BGJXPnY51ydsIcHQl6Gog3WKahjl8C4S1jZMBpbcAhnjkoiOKh98JTUNPLhjhKO1jZR0+SgptHB9qJqNuVXkhwdyu1nDueG2YPbhV1xdSO/fGs7b205zPCkSH6wcAznjh10LNSMMTy/Lp9fvLkdg6GxxYUInDUqmaUzMhmfFkN6XDhh9q4DpHUfNU0Owu1B2IM6h+VfP93Pr97eQZjdxmvfmse4DuM0j320l9++t4vUmDAuy07n8qkZnbZpteNwNf/3zk7qmhzcec4IzhmT0uUBektBJb95dyef7y0jIiSI+mYnowdFcf+isceeU9PYws4jNfzmnZ2sP1TBLy6dwM1zhrb7mz75yT7+37u7ePi6bC7Lzuj2d9DR5vxKvvGv9RzpYkbbM0clcduZwzlrVFKvwiWvrI7EqNBuT39ua29JLYerGpg7IqnfWzOr9x6lrtnJBeM7h35f8FYQBAO7gfOAQmAdcIMxZluH7WKBA0CWMabuZPvVIBiY3s09zBOr9pGZEEF2ZhyTM2OZmBHb7uD8t88O8Mu3tvPcrbOY51wHW56HPe9DSz1VJoL1rtEU2YeQNGwSE6bMJLc5nRe2VloT5rn/mYcE24gJCyYpKpQbzxjC0umZJzxYf7K7lJ+8nsvBsnriI+xcOiWdiyam8vfPD/L+9mLmjUzkoaVTaHEYXlqfz0s5Be0OYinRoaTGhhFsE4Jsgk2EFqeL0tomSmuaaGxxMSQxghdun0NqbNix5+0vrWXRw58yc2gCu4trCA8J4o0751tnZAGvrC/gnpc2c97YFERg1a5SHC7D6EFRXDB+EOeOTSE7K56axhb+8P5u/rU2j5hwO9FhweSXNzB9SDzfv3AMY1OjyS2qIrewmnUHy/lwZwkJkSF8+5yR3DB7MB/tLOE37+7kYFk949NiqGlqIb/cCuqQIBt/uDabJZPT6MjpMlzx+OccqWpk5T1nEx1mb7feGNPpYP7axgLue2UrKdGhPHXzDDLiwimra6KsrpkvD5Tzz9UHKalpYmxqNIsmpmEwtDhdNDtcjE2N4fKpGe0O3o0tTv53+Q6eWZNHdFgw188azFfnDiWjQ6utpLqRNzYX8fqmIrYWVgEwPCmSby4YweVTM7oM6b7kcLr43fu7eWLVPkTg8RumsWhS59/p6fJKELhfeDHwR6zTR582xvxaRO4AMMY86d7ma1hjCdf1ZJ8aBIGrscXJuQ+tIiUmjNe+NRcRoaSsgv977AkuCsphbkQB4VX7sdNy7DlHSKI+dhTxQyYSHRVFcOuBwhbkvqgt1bqwLTrVuuAtOKTT6zqcLj7dc5RXNhSwYnsxzQ4XIUE2frBwDP81b1i7gVGny7DxUAV5ZfUUVDRQWFlPcXUTTpexHsZgDxKSo0JJjg4lLiKEJ1btIy02jJfumENcRAhOl+HaP69hd3ENH/z32eRX1HPdU2uZPzKJv311Jp/vO8otf1/H7OEJ/P1rswgJtlFe18zbW4p4c8th1udV4HQZ4iPsGKC6oYUbZw/hngutLr6Xcgp4ZOWeTp+6M+PDuXJqBl8/a3i7A3eL08WyLw/xyoZCMuPCGZcWzbi0GCZnxpEc3f1Yzab8Sq54/HNumz+M/1ly/ITBnIPlfPO5DThdhlEpUYweFE2zw8ULOfmcMTyBx2+cTkJk579Dk8PJm5sP89dP9x87+aA1YJscLkalRHHfwrGcNy6FvSW13LVsIzuP1PCVOUMor2vmnVxrltxzx6YQJMLhqgYKKxs5WtsEwKSMWC7LTic5OpQ/f7yf7YeryYgL5+tnDuOKqZnHQvhESmuaiAkP7rYrsaPi6kbu+vdGvjxYzvWzsth1pIZtRdX8++tnMH1IfI/20VNeCwJP0CAIbC+sO8R9r2zlqZunc964Qdz01y/YlF/JG9+ex6hB0eByUpK/my0bv2AkBQx25mEr3WFNdeFyHN+RywF08W8/ItEKiPihkDULsmZD+tRj1zJUNbTw8e5SxqVGW6/XB1bvO8rX/r6O8WkxPHfbbF5Yl88v3trO75ZO4arp1iD9v9Yc5Mevb2Pp9EzeyT1CZnw4L94xh5iwzgenqvoWPtlTyoc7S6hvdnD3eaMZn96+y6ixxcnL6wuobXIwKSOWCekxxEV0Pviergde3cKLOQW8c/eZjB4UzXvbjvCdZRtJjwtn9jCrtbOnuJaaJgc3nzGEn1wy/qSfwI0xNDtd2G02bDbBGMO7uUf4f+/t4sDROqZkxbHrSDWRIcE8dM0UzhmTAkBhZQP/XH2QNzcXERkaTFpsGOmx4QxOjOCiCamMTIlq9xof7Srh0Q/3suFQJaHBNhZNTOXamYM5Y3hCp9aMMYYnPra6w2LDrZbj1dMzmZxpneV2uKqRPSW1HCqro7rRQW2Tg7omB29vOUxDi5P/vWISl0/NoLyumSsf/5zqRgevfnNuu5MVmh0ump2uHnVxdUWDQA0YDqeLC/7wCSFBNi6aMIhHPtzLb6+ezNIZWb3bkcsJ9WVQcxiqD1tfa0ug9gjUFMPRXVC219rWZrfCYNiZMHQ+ZJ3R59NhvLftCN98dj3Th8SztbCKuSOS+NtXZxw74BhjuOelzby6oZC02DBe/dZc0mJ9/0K78rpmzv3dKsamRnPx5HR+8noukzPjePprM4996jfGUNfsPOUDXKsWp4vn1+XzyMo9jEuL4aGlk0mJDjv5E08it7CKF9bl859NhdQ0OhibGs29F43h3LHWmEmzwzpz6uX1BSyckEpIsI33th2hyeEiIy6cqoYWapsc7fYZbBOiwoIZkRzFb66axMiU4x8qDh6t44rHPyc23M6PloxnS0ElXx4sZ1N+JXecPYLvnj/6lN6HBoEaUN7cXMRdyzYCcOW0DH5/TbZnXqjuKOR/CflrIW81FG4A47SCIXWiFQ5p2ZCeDSkTTvkMplYv5eRz78tbiA4L5v3vnd1uzACsT/F/+nAPV0zNaHfg8HXPrs3jR//JBaxumUdvmEpESB9cwuRyQuUh6+r0xspjExsaDGILgqBQCAqBIDtEJlsXKYbHn/LUJQ3NTt7eephHP9zDwbJ6Zg6N51vnjOSJVfv48kA53z1/FHefNwoRobqxhbe3HObjXaWkxoYxMiWKUSlRDEuKJCbcTmiw7YQD3uvzKrjhL2tpcriwCYxPj2Hm0AQWTkhl9vDOkz/2hAaBGlBcLsMVT6ymodnBa9+a1+tTW09ZUw0c+gIOfgqF6+HwFmiyBhcJiYbBs2HIPKvVkDoZ7L38NOp08OkXX5BiShgT3WIFUUM5SBCExUBojPU1PN7qwgpPsL52Ma7hS5wuwzefXU9abBg/XjSS4MYKqCu1ZqVtqobGaut321IPjiZwNICj2Tpgi4DYrIN+Y5X7UWm12ioOgLO5d8UEh1ldfxEJ1sWJYbEQGg22YOt1xGZ9HxxqbRscBpFJkDoJkseBPYwWp4sX1uXz8Mo9lNY0ERJs47dXT+ayKelQXw7NteBsAWeT1QUZEuV+nZhe/a1yC6sor2tm2gkmfOwNDQI14DS2OBGhx4NyHuFyWQejoo1WiyHvcyjdaa2z2WHQBMiYZrUcUsZD8hjroAPQXA/FudZzD2+B4q3WBH3OpvavITYwLk4oIgliMyAmAxKGw8jzYMj8/gsIZwsc3gyHN0H5ASjfbz0aqwD3wRyB5hr3spOw2a0DsTGAsd6/2I4fuMNirU/4iSMgcSQkjrICsfV1wDoAO5ut2hyNVvDUHIbqQqg5Ag2Vx4OlqcZq6RmX9XA6rL+Do8MprBJk/Q1jsyAkAkdQOHsrnKQG1xLXkG+99+YaTsgeYQV5WJz1NTTaarEE2a33bQs+Hn5tH7Yg6+vwBTD6ot79fVrL1yBQqp/UHYVDa6wWQ+EGKNp0vNUA7oNIFBzdbR18wDqoDZpoBUfqJGugOiLJ/ak1DjDWwar103NDhTW+0fqoLoTqIqgqhPJ91gEsNMYKhBHnWQevxJHW/oyBqgIrhIpzrefUl1n7bKi0Pv0mj7GmBk8aZR24jh1gDTTXuWupgap8KwAPfQEt7jO/g8MhYZgVSOHx1nMM1teQSOt9RbofYXFtWjqx1oB8cJh10PMFLpcVJtWFcGTr8UftESvIW+qt30dEgjVJYsJw628XFmN1SwWHWOHRXNe+NdNQ6f5aYf09XS1WYLlarJaPcYefaf3eadViXDDnW3DOD0/p7WgQKOUtra2G0p3WndtKd1oH0dRJVkshfap1+mpfTbnd0gD7V8HOt2H3u9Yn4Vbh8daBpbHy+LKIJOvTdGvo1B6xbjnactJLeizJ46yusKHzIHMmRKeDTe935Ys0CJQKRC6Xu5tmnzXNd9keQKyB7kGTYND4411VHZ9X3dq6aOZY9wwCoVHWc0Ki3OMUcf37ntQpO1EQ6I1plBqobDZIGmk9etOvbLNBXJb1UAFB23BKKRXgNAiUUirAaRAopVSA0yBQSqkAp0GglFIBToNAKaUCnAaBUkoFOA0CpZQKcH53ZbGIlAJ5p/j0JOBoH5bTl3y1Nl+tC7S2U+GrdYHv1uardUHvahtijEnuaoXfBcHpEJGc7i6x9jZfrc1X6wKt7VT4al3gu7X5al3Qd7Vp15BSSgU4DQKllApwgRYET3m7gBPw1dp8tS7Q2k6Fr9YFvlubr9YFfVRbQI0RKKWU6izQWgRKKaU60CBQSqkAFzBBICILRWSXiOwVkfu9XMvTIlIiIrltliWIyPsissf9Nd4LdWWJyEciskNEtonI3b5Qm4iEiciXIrLZXdfPfaGuDjUGichGEXnLV2oTkYMislVENolIjq/U5a4jTkReFpGd7n9vc3yhNhEZ4/59tT6qReS7PlLb99z//nNFZJn7/0Wf1BUQQSAiQcBjwCJgPHC9iIz3Ykn/ABZ2WHY/sNIYMwpY6f65vzmAe4wx44AzgDvdvydv19YEnGuMmQJkAwtF5AwfqKutu4EdbX72ldrOMcZktznX3Ffqehh41xgzFpiC9bvzem3GmF3u31c2MB2oB17zdm0ikgF8B5hhjJkIBAHX9VldxpgB/wDmAO+1+fkB4AEv1zQUyG3z8y4gzf19GrDLB35vrwMX+FJtQASwAZjtK3UBme7/hOcCb/nK3xM4CCR1WOYLdcUAB3CfrOJLtXWo50Lgc1+oDcgA8oEErFsMv+Wur0/qCogWAcd/ia0K3Mt8ySBjzGEA99cUbxYjIkOBqcAX+EBt7q6XTUAJ8L4xxifqcvsj8APA1WaZL9RmgBUisl5EbvehuoYDpcDf3d1pfxWRSB+pra3rgGXu771amzGmEHgIOAQcBqqMMSv6qq5ACQLpYpmeN9sNEYkCXgG+a4yp9nY9AMYYp7Ga65nALBGZ6OWSABCRi4ESY8x6b9fShXnGmGlYXaJ3ishZ3i7ILRiYBjxhjJkK1OHdbr1ORCQEuBR4ydu1ALj7/i8DhgHpQKSI3NRX+w+UICgAstr8nAkUeamW7hSLSBqA+2uJN4oQETtWCDxnjHnVl2oDMMZUAquwxlh8oa55wKUichB4HjhXRJ71hdqMMUXuryVY/dyzfKEurP+PBe5WHcDLWMHgC7W1WgRsMMYUu3/2dm3nAweMMaXGmBbgVWBuX9UVKEGwDhglIsPcSX8d8IaXa+roDeCr7u+/itU/369ERIC/ATuMMb/3ldpEJFlE4tzfh2P9p9jp7boAjDEPGGMyjTFDsf5dfWiMucnbtYlIpIhEt36P1Z+c6+26AIwxR4B8ERnjXnQesN0Xamvjeo53C4H3azsEnCEiEe7/p+dhDbD3TV3eHIzp58GWxcBuYB/wP16uZRlWP18L1qejW4FErAHHPe6vCV6oaz5Wl9kWYJP7sdjbtQGTgY3uunKBn7iXe/131qHOBRwfLPb272w4sNn92Nb6b97bdbWpLxvIcf9N/wPE+1BtEUAZENtmmddrA36O9QEoF/gXENpXdekUE0opFeACpWtIKaVUNzQIlFIqwGkQKKVUgNMgUEqpAKdBoJRSAU6DQKl+JCILWmcoVcpXaBAopVSA0yBQqgsicpP7HgibROTP7knvakXkdyKyQURWikiye9tsEVkrIltE5LXWOeFFZKSIfCDWfRQ2iMgI9+6j2szF/5z7SlGlvEaDQKkORGQccC3WpG3ZgBO4EYjEmn9mGvAx8FP3U54B7jPGTAa2tln+HPCYse6jMBfranKwZnX9Lta9MYZjzVeklNcEe7sApXzQeVg3JVnn/rAejjWZlwt4wb3Ns8CrIhILxBljPnYv/yfwknuenwxjzGsAxphGAPf+vjTGFLh/3oR1b4rPPP6ulOqGBoFSnQnwT2PMA+0Wivy4w3Ynmp/lRN09TW2+d6L/D5WXadeQUp2tBK4WkRQ4dp/fIVj/X652b3MD8JkxpgqoEJEz3ctvBj421n0cCkTkcvc+QkUkoj/fhFI9pZ9ElOrAGLNdRH6EdXcvG9YssXdi3UBlgoisB6qwxhHAmv73SfeBfj9wi3v5zcCfReQX7n0s7ce3oVSP6eyjSvWQiNQaY6K8XYdSfU27hpRSKsBpi0AppQKctgiUUirAaRAopVSA0yBQSqkAp0GglFIBToNAKaUC3P8HiDQwDts3J3kAAAAASUVORK5CYII=\n",
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
    "\n",
    "# summarize history for loss\n",
    "plt.plot(history.history['main_output_loss'])\n",
    "plt.plot(history.history['val_main_output_loss'])\n",
    "plt.title('main model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'test'], loc='upper left')\n",
    "plt.show()\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "182ff91a",
   "metadata": {},
   "outputs": [],
   "source": [
    "y_score = model.predict({'main_input': dframe_test_X_time.astype(np.float32), 'aux_input': dframe_test_static.astype(np.float32)})\n",
    "y_pred = y_score"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "c4877ad8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import roc_curve, auc\n",
    "from sklearn import metrics\n",
    "def compute_roc(y_test,y_score,method):\n",
    "    fpr = dict()\n",
    "    tpr = dict()\n",
    "    roc_auc = dict()\n",
    "    n_classes = 2\n",
    "    fpr, tpr, _ = metrics.roc_curve(y_test.ravel(), y_score.ravel())\n",
    "    # Compute micro-average ROC curve and ROC area\n",
    "    roc_auc = auc(fpr, tpr)\n",
    "    plt.figure()\n",
    "    plt.plot(fpr, tpr, label=method+' (AUC = %0.4f)' % roc_auc)\n",
    "    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')\n",
    "    plt.xlim([0.0, 1.0])\n",
    "    plt.ylim([0.0, 1.05])\n",
    "    plt.xlabel('False Positive Rate')\n",
    "    plt.ylabel('True Positive Rate')\n",
    "    plt.title('Receiver operating characteristic example')\n",
    "    plt.legend(loc=\"lower right\")\n",
    "    plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "2e041faa",
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
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.96      0.68      0.80       120\n",
      "           1       0.16      0.70      0.25        10\n",
      "\n",
      "    accuracy                           0.68       130\n",
      "   macro avg       0.56      0.69      0.53       130\n",
      "weighted avg       0.90      0.68      0.76       130\n",
      "\n"
     ]
    }
   ],
   "source": [
    "# main auc\n",
    "compute_roc(dframe_test_Y,y_score[0],'deep learning')\n",
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(dframe_test_Y, (y_pred[0]>0.5).astype(int)))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "a3f8270c",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model: \"sequential\"\n",
      "_________________________________________________________________\n",
      " Layer (type)                Output Shape              Param #   \n",
      "=================================================================\n",
      " conv1d (Conv1D)             (None, 13, 64)            576       \n",
      "                                                                 \n",
      " max_pooling1d (MaxPooling1D  (None, 6, 64)            0         \n",
      " )                                                               \n",
      "                                                                 \n",
      " flatten (Flatten)           (None, 384)               0         \n",
      "                                                                 \n",
      " main_output (Dense)         (None, 1)                 385       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 961\n",
      "Trainable params: 961\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n",
      "Epoch 1/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 1.3784\n",
      "Epoch 2/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 1.1351\n",
      "Epoch 3/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 1.0315\n",
      "Epoch 4/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9957\n",
      "Epoch 5/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9764\n",
      "Epoch 6/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9577\n",
      "Epoch 7/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9429\n",
      "Epoch 8/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9302\n",
      "Epoch 9/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9177\n",
      "Epoch 10/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9109\n",
      "Epoch 11/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9050\n",
      "Epoch 12/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.9002\n",
      "Epoch 13/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8935\n",
      "Epoch 14/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8874\n",
      "Epoch 15/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8821\n",
      "Epoch 16/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8768\n",
      "Epoch 17/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8719\n",
      "Epoch 18/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8680\n",
      "Epoch 19/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8630\n",
      "Epoch 20/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8583\n",
      "Epoch 21/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8530\n",
      "Epoch 22/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8477\n",
      "Epoch 23/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8436\n",
      "Epoch 24/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.8415\n",
      "Epoch 25/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8351\n",
      "Epoch 26/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8316\n",
      "Epoch 27/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.8309\n",
      "Epoch 28/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8225\n",
      "Epoch 29/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8190\n",
      "Epoch 30/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.8130\n",
      "Epoch 31/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8063\n",
      "Epoch 32/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.8037\n",
      "Epoch 33/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7987\n",
      "Epoch 34/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7943\n",
      "Epoch 35/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7894\n",
      "Epoch 36/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7869\n",
      "Epoch 37/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7835\n",
      "Epoch 38/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7792\n",
      "Epoch 39/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7756\n",
      "Epoch 40/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7708\n",
      "Epoch 41/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7676\n",
      "Epoch 42/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7635\n",
      "Epoch 43/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7586\n",
      "Epoch 44/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7578\n",
      "Epoch 45/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7529\n",
      "Epoch 46/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7486\n",
      "Epoch 47/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7455\n",
      "Epoch 48/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.7396\n",
      "Epoch 49/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.7378\n",
      "Epoch 50/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.7333\n",
      "Epoch 51/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7294\n",
      "Epoch 52/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7269\n",
      "Epoch 53/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7238\n",
      "Epoch 54/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7234\n",
      "Epoch 55/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7173\n",
      "Epoch 56/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7147\n",
      "Epoch 57/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.7124\n",
      "Epoch 58/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7137\n",
      "Epoch 59/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7046\n",
      "Epoch 60/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7027\n",
      "Epoch 61/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.7001\n",
      "Epoch 62/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6989\n",
      "Epoch 63/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6969\n",
      "Epoch 64/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6953\n",
      "Epoch 65/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6907\n",
      "Epoch 66/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6893\n",
      "Epoch 67/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6853\n",
      "Epoch 68/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6811\n",
      "Epoch 69/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6818\n",
      "Epoch 70/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6806\n",
      "Epoch 71/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6746\n",
      "Epoch 72/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6707\n",
      "Epoch 73/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6711\n",
      "Epoch 74/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6745\n",
      "Epoch 75/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6672\n",
      "Epoch 76/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6657\n",
      "Epoch 77/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6617\n",
      "Epoch 78/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6600\n",
      "Epoch 79/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6596\n",
      "Epoch 80/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6543\n",
      "Epoch 81/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.6526\n",
      "Epoch 82/100\n",
      "6/6 [==============================] - 0s 3ms/step - loss: 0.6508\n",
      "Epoch 83/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6482\n",
      "Epoch 84/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6461\n",
      "Epoch 85/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6454\n",
      "Epoch 86/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6439\n",
      "Epoch 87/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6396\n",
      "Epoch 88/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6363\n",
      "Epoch 89/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6364\n",
      "Epoch 90/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6326\n",
      "Epoch 91/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6294\n",
      "Epoch 92/100\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6370\n",
      "Epoch 93/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6268\n",
      "Epoch 94/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6271\n",
      "Epoch 95/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6258\n",
      "Epoch 96/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6217\n",
      "Epoch 97/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6161\n",
      "Epoch 98/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6207\n",
      "Epoch 99/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6192\n",
      "Epoch 100/100\n",
      "6/6 [==============================] - 0s 2ms/step - loss: 0.6113\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "<keras.callbacks.History at 0x2880b2901c0>"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "from keras.models import Sequential\n",
    "from keras.layers import Dense\n",
    "from keras.layers import Flatten\n",
    "from keras.layers.convolutional import Conv1D\n",
    "from keras.layers.convolutional import MaxPooling1D\n",
    "\n",
    "\n",
    "# define model\n",
    "model = Sequential()\n",
    "model.add(Conv1D(filters=64, kernel_size=2, activation='relu', input_shape=(dframe_train_X_time.shape[1], dframe_train_X_time.shape[2])))\n",
    "model.add(MaxPooling1D(pool_size=2))\n",
    "model.add(Flatten())\n",
    "model.add( Dense(1, activation='sigmoid', name='main_output'))\n",
    "          \n",
    "model.summary()\n",
    "model.compile(optimizer='adam', loss='binary_crossentropy')\n",
    "\n",
    "model.fit(dframe_train_X_time.astype(np.float32), dframe_train_Y.astype(np.float32), epochs=100, class_weight={0:1,1:2.5})"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "6baa5414",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "              precision    recall  f1-score   support\n",
      "\n",
      "           0       0.97      0.32      0.48       120\n",
      "           1       0.10      0.90      0.18        10\n",
      "\n",
      "    accuracy                           0.36       130\n",
      "   macro avg       0.54      0.61      0.33       130\n",
      "weighted avg       0.91      0.36      0.45       130\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "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/LGLKsV9mNw3mDKF87NixUEGD17Ivn3n6N27Fo0bB7g6VJUDbdy4kVmzZrF582YSExNp0KBBSqIYMGAAEydOpHLlyqxbt44hQ4bw559/Mnz4cJ577jmaN2/O0aNHadeuHbt27QJg27ZtrF27losXL1K/fn06duyYcoK53tKlS9m3bx/r16/HGEOnTp1YvXo1LVu25Ntvv6Vw4cJcunSJhg0b0r17d/z9/bl48SK1atXinXfeAawTVZMmTXjvvfd46aWXmDx5Mq+//voN20pMTGT9+vUsWrSIt99+m+XLlzN+/HgKFSrEtm3bCA0NpV69eqnGuWbNmmtOvGvWrCEwMJCKFSvSqlUrFi1aRLdu3dI8ztHR0URHR1OxYsV0/yejR49mxowZN7zfsmVLvvzyy2veO378OGXKXL2dQEBAAMePH79h2b179wLQrFkzkpKSGDVqFO3bt09z+SlTptChQwe8vLzw9fVl7dq1KfPNnTuXkSNHcubMGRYuXJjyfq1atdiwYUO6+5gRck2iiEtIYuSv25m7+TgdapdgTI+65M/rzrPPLuaLL9ZRqVJhli59lDZt0v9wqewvvV/+zvD333/TtWtX8ufPD0CnTp0AiImJ4d9//6VHjx4p88bHxwOwfPlydu7cmfJ+VFQU0dHRAHTu3BkvLy+8vLxo3bo169evp0uXLqlue+nSpSxdupT69eunbHPfvn0pJ8S5c+cCcOzYMfbt24e/vz9ubm507949ZR158+bl/vvvByAoKIhly5aluq0rJ/KgoCAOHz4MwD///MPw4cMB6wRXp06dVJc9efIkRYsWTXk9c+ZMevXqBUCvXr2YNm0a3bp1u2lboYik3IveESNGjGDEiBEOzWtSuSVDattJTExk3759rFq1irCwMFq0aEFoaGiay3/22WcsWrSIxo0bM3r0aJ5//nmmTJkCQNeuXenatSurV6/mjTfeYPny5QC4ubmRN29eoqOj8fHxcWgfbleuSBSnIuMYOC2ErWGRvNCmCkNaXU0GjRqV5s03WzJyZAs8PXPF4VAulNqJJTk5mYIFC7Jly5ZUp/333394eXmlu660To7GGEaOHMnAgQOveX/VqlUsX76c//77j/z589OqVauUnuuenp64uV29BNzDwyNlG25ubjdtSM2XL98N86R2kkyNl5dXyvaTkpL45Zdf+O2333jvvfdSOpNFR0fj7+/P+fPnr1n23LlzBAYG4uvrS4ECBTh48CAVKlRIc3u3UqIICAhg1apVKa/DwsJo1arVDcsGBATQpEkTPDw8CAwMpGrVquzbt++my4eHh7N161YaN24MQM+ePWnfvn2qMR04cICzZ89SpEgRwPpB4UhV3J3K8Vc9bTp6ngfG/sP+MzFM6hNEyyI+NGv2LePGWUW2hx+uzdtvt9YkoZyuZcuWzJ07l0uXLhEdHc3vv/8OgK+vL4GBgfz888+AdVLdunUrAG3btmXs2LEp67BPJvPnzycuLo6IiAhWrVpFw4YNb7rtdu3a8e233xITEwNY1ShnzpwhMjKSQoUKkT9/fnbv3n1NlUdGat68ObNnzwZg586dbN++PdX5qlevnnLFz/Lly6lbty7Hjh3j8OHDHDlyhO7duzNv3jy8vb0pWbIkK1asAKwksXjxYpo3bw7AyJEjGTp0aEq9f1RUFJMmTbpheyNGjGDLli03PK5PEmAdw6VLl3L+/HnOnz/P0qVLadeu3Q3zdenShZUrVwJw9uxZ9u7dS4UKFW66fKFChYiMjEypslq2bBnVq1cHYP/+/SlJdtOmTVy+fBl/f38AIiIiKFq0aIYO1XEzOfrsODvkGK/PDaVkQU8m9a7P9HEhfPHFOgoX9qJECedfNqiUvQYNGtCzZ0/q1atHuXLlaNGiRcq0GTNmMHjwYN59910SEhLo1asXdevW5csvv2To0KHUqVOHxMREWrZsycSJEwFo1KgRHTt25OjRo7zxxhs3bZ8AK+Hs2rWLu+66C7AaSadPn0779u2ZOHEiderUoWrVqjRp0sQp+z5kyBAee+wx6tSpQ/369alTpw5+fn43zNexY0dWrVrFvffey8yZM+nates107t3786ECRPo06cPU6dOZejQobzwwgsAvPXWWyntEoMHDyYmJoaGDRvi4eGBh4dHyny3q3DhwrzxxhspCfnNN99MaZh+8803CQ4OplOnTikJoUaNGri5uTF69OiUk/vNlp88eTLdu3cnT548FCpUiG+//RaAX375halTp+Lh4YGXlxc//fRTSqlu5cqVdOjQ4Y72yVE58laoiUnJvLdoF9+tOUzzSkV4sERBnh68iLCwKAYMaMCHH95LoUI3FuVVzrZr166UX2rZnf2VRtlBUlISCQkJeHp6cuDAAe655x727t1L3rx5r5nv0qVLtG7dmjVr1lxT7aVu1K1bNz744AOqVq16w7TUPut3civUHFeiOH/xMsNmbmLN/gieaBbIqx2q8e+aYxQu7MVPPz1I06Zl0l+JUipDxcbG0rp1axISEjDGMGHChBuSBFhtFG+//TbHjx+nbFkdbPNmLl++TJcuXVJNEs6Qo0oUe09H89QPIZw8F0uTS0I5H0/effduAJKTDXnyaK/q3CwnlSiUSouWKG5iyY5TPP/TFpJPx8LqE0zfHUG3btVTEoQmCQXc0qWTSmVHzvjxn+0TRXKy4as/9zPm9524hZzhyL8nKFPGl3nzetK5czVXh6eyEE9PTyIiIvD399dkoXKkK5cQZ/Qls9k6UVyMT+TFn7fyR+gp7g305+dvdvDii3fx1lut8Pa+sf5T5W4BAQGEhYVl+Fj9SmUlV+5wl5GybaI4di6Wh8esJvTvY4x+/x6ebB7ImD4N8PfP7+rQVBZ1pQOUUurWOLXDnYi0F5E9IrJfRF5JZbqIyJe26dtEpIEj61214zQNu/3Imo/Xk7gtgnbl/W13dtIkoZRSGc1piUJE3IBxwH1ADaC3iNS4brb7gMq2xwBgQnrrDTsdQ5vm33H6r+N06lqN/XuHUabMjR13lFJKZQxnVj01AvYbYw4CiMgsoDOw026ezsBU21jpa0WkoIiUNMacvNlKT5+IxreoG9MX9eaB+6o4MXyllFLg3ERRGjhm9zoMaOzAPKWBaxKFiAzAKnEAxEedfim0U4eXMjba7KkIcNbVQWQReiyu0mNxlR6Lq267d54zE0Vq1x9ef4GvI/NgjJkETAIQkZDb7TSS0+ixuEqPxVV6LK7SY3GViKQ99lEanNmYHQbYj5cRAJy4jXmUUkq5kDMTxQagsogEikheoBfw23Xz/Ab0tV391ASITKt9QimlVOZzWtWTMSZRRIYBSwA34FtjzA4RGWSbPhFYBHQA9gOxwOMOrPrGQeVzLz0WV+mxuEqPxVV6LK667WOR7QYFVEoplbly/B3ulFJK3RlNFEoppdKUZROFs4b/yI4cOBaP2I7BNhH5V0TquiLOzJDesbCbr6GIJInIg5kZX2Zy5FiISCsR2SIiO0Tkr8yOMbM48B3xE5HfRWSr7Vg40h6a7YjItyJyRkRCbzL99s6bxpgs98Bq/D4AVADyAluBGtfN0wH4A6svRhNgnavjduGxaAoUsj2/LzcfC7v5/sS6WOJBV8ftws9FQayREMraXhdzddwuPBavAh/ZnhcFzgF5XR27E45FS6ABEHqT6bd13syqJYqU4T+MMZeBK8N/2EsZ/sMYsxYoKCIlMzvQTJDusTDG/GuMOW97uRarP0pO5MjnAuBp4BfgTGYGl8kcORYPA78aY44CGGNy6vFw5FgYwEesG5F4YyWKxMwN0/mMMaux9u1mbuu8mVUTxc2G9rjVeXKCW93PJ7F+MeRE6R4LESkNdAUmZmJcruDI56IKUEhEVonIRhHpm2nRZS5HjsVYoDpWh97twHBjTHLmhJel3NZ5M6vejyLDhv/IARzeTxFpjZUomjs1Itdx5Fh8DrxsjEnK4Xexc+RYuANBwD2AF/CfiKw1xux1dnCZzJFj0Q7YAtwNVASWicjfxpgoJ8eW1dzWeTOrJgod/uMqh/ZTROoAU4D7jDERmRRbZnPkWAQDs2xJogjQQUQSjTHzMiXCzOPod+SsMeYicFFEVgN1gZyWKBw5Fo8DHxqron6/iBwCqgHrMyfELOO2zptZtepJh/+4Kt1jISJlgV+BPjnw16K9dI+FMSbQGFPeGFMemAMMyYFJAhz7jswHWoiIu4jkxxq9eVcmx5kZHDkWR7FKVohIcayRVA9mapRZw22dN7NkicI4b/iPbMfBY/Em4A+Mt/2STjQ5cMRMB49FruDIsTDG7BKRxcA2IBmYYoxJ9bLJ7MzBz8X/gO9FZDtW9cvLxpgcN/y4iMwEWgFFRCQMeAvwgDs7b+oQHkoppdKUVauelFJKZRGaKJRSSqVJE4VSSqk0aaJQSimVJk0USiml0qSJQmVJtpFft9g9yqcxb0wGbO97ETlk29YmEbnrNtYxRURq2J6/et20f+80Rtt6rhyXUNtoqAXTmb+eiHTIiG2r3Esvj1VZkojEGGO8M3reNNbxPbDAGDNHRNoCY4wxde5gfXccU3rrFZEfgL3GmPfSmL8fEGyMGZbRsajcQ0sUKlsQEW8RWWH7tb9dRG4YNVZESorIartf3C1s77cVkf9sy/4sIumdwFcDlWzLPm9bV6iIPGt7r4CILLTd2yBURHra3l8lIsEi8iHgZYtjhm1ajO3vT/a/8G0lme4i4iYio0Vkg1j3CRjowGH5D9uAbiLSSKx7kWy2/a1q66X8DtDTFktPW+zf2razObXjqNQNXD1+uj70kdoDSMIaxG0LMBdrFAFf27QiWD1Lr5SIY2x/XwBesz13A3xs864GCtjefxl4M5XtfY/t3hVAD2Ad1oB624ECWENT7wDqA92ByXbL+tn+rsL69Z4Sk908V2LsCvxge54XayRPL2AA8Lrt/XxACBCYSpwxdvv3M9De9toXcLc9vxf4xfa8HzDWbvn3gUdtzwtijftUwNX/b31k7UeWHMJDKeCSMabelRci4gG8LyItsYajKA0UB07ZLbMB+NY27zxjzBYR+T+gBrDGNrxJXqxf4qkZLSKvA+FYo/DeA8w11qB6iMivQAtgMTBGRD7Cqq76+xb26w/gSxHJB7QHVhtjLtmqu+rI1Tvy+QGVgUPXLe8lIluA8sBGYJnd/D+ISGWs0UA9brL9tkAnEXnR9toTKEvOHANKZRBNFCq7eATrzmRBxpgEETmMdZJLYYxZbUskHYFpIjIaOA8sM8b0dmAbI4wxc668EJF7U5vJGLNXRIKwxsz5QESWGmPecWQnjDFxIrIKa9jrnsDMK5sDnjbGLElnFZeMMfVExA9YAAwFvsQay2ilMaarreF/1U2WF6C7MWaPI/EqBdpGobIPP+CMLUm0BspdP4OIlLPNMxn4BuuWkGuBZiJypc0hv4hUcXCbq4EutmUKYFUb/S0ipYBYY8x0YIxtO9dLsJVsUjMLazC2FlgD2WH7O/jKMiJSxbbNVBljIoFngBdty/gBx22T+9nNGo1VBXfFEuBpsRWvRKT+zbah1BWaKFR2MQMIFpEQrNLF7lTmaQVsEZHNWO0IXxhjwrFOnDNFZBtW4qjmyAaNMZuw2i7WY7VZTDHGbAZqA+ttVUCvAe+msvgkYNuVxuzrLMW6t/FyY926E6x7iewENolIKPA16ZT4bbFsxRpW+2Os0s0arPaLK1YCNa40ZmOVPDxssYXaXiuVJr08VimlVJq0RKGUUipNmiiUUkqlSROFUkqpNGmiUEoplSZNFEoppdKkiUIppVSaNFEopZRK0/8D9Z0esL+lLyMAAAAASUVORK5CYII=\n",
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
    "from sklearn.metrics import classification_report\n",
    "y_pred=model.predict(dframe_test_X_time, verbose=0)\n",
    "print(classification_report(dframe_test_Y, (y_pred>0.5).astype(int)))\n",
    "compute_roc(dframe_test_Y,(y_pred>0.5).astype(int),'deep learning')"
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
