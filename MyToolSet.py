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
   

###################################################################################################

{
 "cells": [
  {
   "cell_type": "markdown",
   "id": "09991d7d",
   "metadata": {},
   "source": [
    "# <font color='Blue'> Analysis of Chronic Kidney Disease Progression in Patients </font>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "14a37ad2",
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
   "id": "700f616e",
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
   "id": "ab6aabee",
   "metadata": {},
   "source": [
    "### 1. Data Import"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "6cffa8ae",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "--------------------------Rows fetched----> 300\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 1439\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 1821\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 1809\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 1556\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 2025\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 1261\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 2181\n",
      "Unique Patients---> 272\n",
      "Lab records per patient--------------\n",
      "Rows fetched----> 300\n",
      "Unique Patients---> 300\n",
      "Lab records per patient--------------\n"
     ]
    }
   ],
   "source": [
    "# --------------------------- Import individual files and process data for quality checks\n",
    "\n",
    "\n",
    "dframe_demographics=pd.read_csv('T_demo.csv')\n",
    "print('--------------------------Rows fetched---->',len(dframe_demographics))\n",
    "print('Unique Patients--->',len(dframe_demographics['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_demographics.groupby('id').size())\n",
    "\n",
    "dframe_creatinine=pd.read_csv('T_creatinine.csv')\n",
    "print('Rows fetched---->',len(dframe_creatinine))\n",
    "print('Unique Patients--->',len(dframe_creatinine['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_creatinine.groupby('id').size())\n",
    "\n",
    "dframe_bloodpressure_d=pd.read_csv('T_DBP.csv')\n",
    "print('Rows fetched---->',len(dframe_bloodpressure_d))\n",
    "print('Unique Patients--->',len(dframe_bloodpressure_d['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_bloodpressure_d.groupby('id').size())\n",
    "\n",
    "dframe_bloodpressure_s=pd.read_csv('T_SBP.csv')\n",
    "print('Rows fetched---->',len(dframe_bloodpressure_s))\n",
    "print('Unique Patients--->',len(dframe_bloodpressure_s['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_bloodpressure_s.groupby('id').size())\n",
    "\n",
    "dframe_bloodsugar=pd.read_csv('T_glucose.csv')\n",
    "print('Rows fetched---->',len(dframe_bloodsugar))\n",
    "print('Unique Patients--->',len(dframe_bloodsugar['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_bloodsugar.groupby('id').size())\n",
    "\n",
    "dframe_hglob=pd.read_csv('T_HGB.csv')\n",
    "print('Rows fetched---->',len(dframe_hglob))\n",
    "print('Unique Patients--->',len(dframe_hglob['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_hglob.groupby('id').size())\n",
    "\n",
    "dframe_lipoprotein=pd.read_csv('T_ldl.csv')\n",
    "print('Rows fetched---->',len(dframe_lipoprotein))\n",
    "print('Unique Patients--->',len(dframe_lipoprotein['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_lipoprotein.groupby('id').size())\n",
    "\n",
    "dframe_medications=pd.read_csv('T_meds.csv')\n",
    "print('Rows fetched---->',len(dframe_medications))\n",
    "print('Unique Patients--->',len(dframe_medications['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_medications.groupby('id').size())\n",
    "\n",
    "dframe_target=pd.read_csv('T_stage.csv')\n",
    "print('Rows fetched---->',len(dframe_target))\n",
    "print('Unique Patients--->',len(dframe_target['id'].unique()))\n",
    "print('Lab records per patient--------------')\n",
    "# print(dframe_target.groupby('id').size())"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "bc3ca4db",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      "\n",
      " All lab test results Dataframe size----> (4761, 8)\n"
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
       "      <th>Creatinine</th>\n",
       "      <th>time</th>\n",
       "      <th>DBP</th>\n",
       "      <th>SBP</th>\n",
       "      <th>Glucose</th>\n",
       "      <th>HGB</th>\n",
       "      <th>Lipoprotein</th>\n",
       "      <th>Time Window</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>1.29</td>\n",
       "      <td>0</td>\n",
       "      <td>95.32</td>\n",
       "      <td>134.11</td>\n",
       "      <td>6.24</td>\n",
       "      <td>13.51</td>\n",
       "      <td>161.49</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1439</th>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>98</td>\n",
       "      <td>83.98</td>\n",
       "      <td>133.75</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>1.15</td>\n",
       "      <td>107</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.20</td>\n",
       "      <td>13.39</td>\n",
       "      <td>111.39</td>\n",
       "      <td>5</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1440</th>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>279</td>\n",
       "      <td>65.97</td>\n",
       "      <td>125.08</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>13</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>1.44</td>\n",
       "      <td>286</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>7.01</td>\n",
       "      <td>12.84</td>\n",
       "      <td>NaN</td>\n",
       "      <td>14</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1441</th>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>380</td>\n",
       "      <td>83.41</td>\n",
       "      <td>136.75</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>1.23</td>\n",
       "      <td>382</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>6.89</td>\n",
       "      <td>13.32</td>\n",
       "      <td>157.90</td>\n",
       "      <td>18</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1442</th>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>475</td>\n",
       "      <td>86.39</td>\n",
       "      <td>130.50</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>23</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>1.08</td>\n",
       "      <td>580</td>\n",
       "      <td>87.64</td>\n",
       "      <td>154.91</td>\n",
       "      <td>5.62</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>28</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1443</th>\n",
       "      <td>0</td>\n",
       "      <td>NaN</td>\n",
       "      <td>678</td>\n",
       "      <td>78.39</td>\n",
       "      <td>154.28</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>NaN</td>\n",
       "      <td>32</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "      id  Creatinine  time    DBP     SBP  Glucose    HGB  Lipoprotein  \\\n",
       "0      0        1.29     0  95.32  134.11     6.24  13.51       161.49   \n",
       "1439   0         NaN    98  83.98  133.75      NaN    NaN          NaN   \n",
       "1      0        1.15   107    NaN     NaN     7.20  13.39       111.39   \n",
       "1440   0         NaN   279  65.97  125.08      NaN    NaN          NaN   \n",
       "2      0        1.44   286    NaN     NaN     7.01  12.84          NaN   \n",
       "1441   0         NaN   380  83.41  136.75      NaN    NaN          NaN   \n",
       "3      0        1.23   382    NaN     NaN     6.89  13.32       157.90   \n",
       "1442   0         NaN   475  86.39  130.50      NaN    NaN          NaN   \n",
       "4      0        1.08   580  87.64  154.91     5.62    NaN          NaN   \n",
       "1443   0         NaN   678  78.39  154.28      NaN    NaN          NaN   \n",
       "\n",
       "      Time Window  \n",
       "0               0  \n",
       "1439            5  \n",
       "1               5  \n",
       "1440           13  \n",
       "2              14  \n",
       "1441           18  \n",
       "3              18  \n",
       "1442           23  \n",
       "4              28  \n",
       "1443           32  "
      ]
     },
     "execution_count": 3,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "## ------------------------------- Merge All test Results Information into one source.\n",
    "\n",
    "from functools import reduce\n",
    "list_df = [dframe_creatinine, dframe_bloodpressure_d, dframe_bloodpressure_s, dframe_bloodsugar, dframe_hglob,dframe_lipoprotein]\n",
    "\n",
    "dframe_all_test_results = list_df[0]\n",
    "for df_ in list_df[1:]:\n",
    "    dframe_all_test_results =pd.DataFrame( dframe_all_test_results.merge(df_,how='outer', on=['id','time']))\n",
    "print('\\n\\n All lab test results Dataframe size---->',dframe_all_test_results.shape)\n",
    "\n",
    "dframe_all_test_results.sort_values(['id','time'],inplace=True)\n",
    "\n",
    "dframe_all_test_results.to_csv( 'output/dframe_all_test_results.csv',index=False)\n",
    "\n",
    "# Smoothen the number of days within a reasonable window of one month to reduce variance of observation \n",
    "\n",
    "  \n",
    "dframe_all_test_results['Time Window']=dframe_all_test_results['time'].apply(lambda x:round(x/21))\n",
    "\n",
    "dframe_all_test_results.head(10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "19a2cd03",
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
    "    else: return 0\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "ea78f7f0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Index(['id', 'Time Window', 'Creatinine_min', 'DBP_min', 'SBP_min',\n",
      "       'Glucose_min', 'HGB_min', 'Lipoprotein_min', 'Creatinine_mean',\n",
      "       'DBP_mean', 'SBP_mean', 'Glucose_mean', 'HGB_mean', 'Lipoprotein_mean',\n",
      "       'Creatinine_max', 'DBP_max', 'SBP_max', 'Glucose_max', 'HGB_max',\n",
      "       'Lipoprotein_max'],\n",
      "      dtype='object')\n"
     ]
    }
   ],
   "source": [
    "##-------------Smoothening the variance caused by days elapsed. Making it a three week aggregation data\n",
    "dframe_smoothened_test_results=pd.DataFrame(dframe_all_test_results.groupby(['id','Time Window'],as_index=False)['time'].nunique())\n",
    "\n",
    "for func in ['min','mean','max']:\n",
    "    dframe_smoothened_test_results_temp=pd.DataFrame(dframe_all_test_results.groupby(['id','Time Window'],as_index=False).agg(func))\n",
    "    dframe_smoothened_test_results_temp.columns=['id','Time Window','Creatinine_'+func,'time_'+func,'DBP_'+func,'SBP_'+func,'Glucose_'+func,'HGB_'+func,'Lipoprotein_'+func]\n",
    "    del(dframe_smoothened_test_results_temp['time_'+func])\n",
    "    dframe_smoothened_test_results=dframe_smoothened_test_results.merge(dframe_smoothened_test_results_temp,on=['id','Time Window'])\n",
    "\n",
    "del(dframe_smoothened_test_results['time'])\n",
    "    \n",
    "                   \n",
    "print(dframe_smoothened_test_results.columns)\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 6,
   "id": "12f7a6b3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "300\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "##-------------------- Imputing non avialable information of lab results on certain days wih previous available data\n",
    "\n",
    "dframe_all_test_results.update(dframe_all_test_results.sort_values([\"id\",\"time\"]).groupby(\"id\").ffill())\n",
    "dframe_smoothened_test_results.update(dframe_smoothened_test_results.sort_values([\"id\",\"Time Window\"]).groupby(\"id\").ffill())\n",
    "\n",
    "dframe_all_test_results.to_csv( 'output/dframe_imputed_all_test_results.csv',index=False)\n",
    "dframe_smoothened_test_results.to_csv( 'output/dframe_smoothened_test_results.csv',index=False)\n",
    "print(len(dframe_all_test_results.id.unique()))\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "5ef17bca",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(4761, 16)\n",
      "300\n",
      "300\n"
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
       "      <th>race</th>\n",
       "      <th>gender</th>\n",
       "      <th>age</th>\n",
       "      <th>Age Group</th>\n",
       "      <th>drug</th>\n",
       "      <th>daily_dosage</th>\n",
       "      <th>start_day</th>\n",
       "      <th>end_day</th>\n",
       "      <th>Medication Duration (days)</th>\n",
       "      <th>Treatment</th>\n",
       "      <th>Stage_Progress</th>\n",
       "      <th>target</th>\n",
       "      <th>CKD(t=0)</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>atorvastatin</td>\n",
       "      <td>10.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>109.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>Medication Period-High Blood Cholestrol</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>atorvastatin</td>\n",
       "      <td>10.0</td>\n",
       "      <td>117.0</td>\n",
       "      <td>207.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>Medication Period-High Blood Cholestrol</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>losartan</td>\n",
       "      <td>100.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>289.0</td>\n",
       "      <td>270.0</td>\n",
       "      <td>Medication Period-Hyper Tension</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>losartan</td>\n",
       "      <td>100.0</td>\n",
       "      <td>403.0</td>\n",
       "      <td>493.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>Medication Period-Hyper Tension</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>losartan</td>\n",
       "      <td>100.0</td>\n",
       "      <td>587.0</td>\n",
       "      <td>677.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>Medication Period-Hyper Tension</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>metformin</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>19.0</td>\n",
       "      <td>109.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>metformin</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>131.0</td>\n",
       "      <td>281.0</td>\n",
       "      <td>150.0</td>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>metformin</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>309.0</td>\n",
       "      <td>399.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>metformin</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>312.0</td>\n",
       "      <td>462.0</td>\n",
       "      <td>150.0</td>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>metformin</td>\n",
       "      <td>1000.0</td>\n",
       "      <td>467.0</td>\n",
       "      <td>557.0</td>\n",
       "      <td>90.0</td>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>True</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "   id     race gender  age Age Group          drug  daily_dosage  start_day  \\\n",
       "0   0  Unknown   Male   70     61-70  atorvastatin          10.0       19.0   \n",
       "1   0  Unknown   Male   70     61-70  atorvastatin          10.0      117.0   \n",
       "2   0  Unknown   Male   70     61-70      losartan         100.0       19.0   \n",
       "3   0  Unknown   Male   70     61-70      losartan         100.0      403.0   \n",
       "4   0  Unknown   Male   70     61-70      losartan         100.0      587.0   \n",
       "5   0  Unknown   Male   70     61-70     metformin        1000.0       19.0   \n",
       "6   0  Unknown   Male   70     61-70     metformin        1000.0      131.0   \n",
       "7   0  Unknown   Male   70     61-70     metformin        1000.0      309.0   \n",
       "8   0  Unknown   Male   70     61-70     metformin        1000.0      312.0   \n",
       "9   0  Unknown   Male   70     61-70     metformin        1000.0      467.0   \n",
       "\n",
       "   end_day  Medication Duration (days)  \\\n",
       "0    109.0                        90.0   \n",
       "1    207.0                        90.0   \n",
       "2    289.0                       270.0   \n",
       "3    493.0                        90.0   \n",
       "4    677.0                        90.0   \n",
       "5    109.0                        90.0   \n",
       "6    281.0                       150.0   \n",
       "7    399.0                        90.0   \n",
       "8    462.0                       150.0   \n",
       "9    557.0                        90.0   \n",
       "\n",
       "                                 Treatment  Stage_Progress  target  CKD(t=0)  \n",
       "0  Medication Period-High Blood Cholestrol            True       1       3.0  \n",
       "1  Medication Period-High Blood Cholestrol            True       1       3.0  \n",
       "2          Medication Period-Hyper Tension            True       1       3.0  \n",
       "3          Medication Period-Hyper Tension            True       1       3.0  \n",
       "4          Medication Period-Hyper Tension            True       1       3.0  \n",
       "5        Medication Period-Diabetes Type 2            True       1       3.0  \n",
       "6        Medication Period-Diabetes Type 2            True       1       3.0  \n",
       "7        Medication Period-Diabetes Type 2            True       1       3.0  \n",
       "8        Medication Period-Diabetes Type 2            True       1       3.0  \n",
       "9        Medication Period-Diabetes Type 2            True       1       3.0  "
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "#------------- Linking patient demographics with medication history and disease outcomes\n",
    "\n",
    "dframe_demo_alltestresults=dframe_demographics.merge(dframe_all_test_results,how='outer',on='id').merge(dframe_target,how='outer',on='id')\n",
    "dframe_demo_alltestresults['eGFR']=dframe_demo_alltestresults.apply(eGFScore,axis=1)\n",
    "\n",
    "print(dframe_demo_alltestresults.shape)\n",
    "print(len(dframe_demo_alltestresults.id.unique()))\n",
    "# Determine whether at t=0 patients have pre-existing conditions.\n",
    "    \n",
    "dframe_demo_alltestresults['CKD(t=0)']=dframe_demo_alltestresults.apply(lambda x: CKD_stage(x['eGFR']) if x['time']==0 else 0 ,axis=1)\n",
    "dframe_demo_alltestresults['Dieabetes(t=0)']=dframe_demo_alltestresults.apply(lambda x: 1 if x['Glucose']>=7.0 and x['time']==0 else 0,axis=1)\n",
    "dframe_demo_alltestresults['Cholestrol(t=0)']=dframe_demo_alltestresults.apply(lambda x: 1 if x['Lipoprotein']>=100 and x['time']==0 else 0,axis=1)\n",
    "dframe_demo_alltestresults['Hyper Tension(t=0)']=dframe_demo_alltestresults.apply(lambda x: 1 if x['SBP']>=130 and x['DBP']>=80 and x['time']==0 else 0,axis=1)\n",
    "dframe_demo_alltestresults['Hemoglobin(t=0)']=dframe_demo_alltestresults.apply(lambda x: 1 if x['HGB']<=(13.2 if x['gender']=='Male' else 11.6)  and x['time']==0 else 0,axis=1)\n",
    "dframe_demo_alltestresults['t=0']=dframe_demo_alltestresults.groupby('id')['time'].transform(min)\n",
    "dframe_demo_alltestresults['t=n']=dframe_demo_alltestresults.groupby('id')['time'].transform(max)\n",
    "dframe_demo_alltestresults['period']=dframe_demo_alltestresults['t=n']-dframe_demo_alltestresults['t=0']\n",
    "# dframe_temp=pd.DataFrame(dframe_demo_alltestresults[dframe_demo_alltestresults.time==0][['id','Creatinine','SBP','HGB','Glucose','Lipoprotein']],columns=['id','Creatinine_baseline','SBP_baseline','HGB_baseline','Glucose_baseline','Lipoprotein_baseline'])\n",
    "# print(dframe_temp.shape)\n",
    "# dframe_demo_alltestresults=dframe_demo_alltestresults.merge(dframe_temp,on='id',how='inner')\n",
    "# print(dframe_demo_alltestresults.shape)\n",
    "# for col in ['Creatinine','SBP','HGB','Glucose','Lipoprotein']:\n",
    "# #     dframe_demo_alltestresults[col+'_next']=dframe_demo_alltestresults.groupby('id')[col].shift(-1)\n",
    "# #     dframe_demo_alltestresults[col+'_delta']=dframe_demo_alltestresults[col+'_next']-dframe_demo_alltestresults[col]\n",
    "    \n",
    "#     dframe_demo_alltestresults[col+'_change from baseline']=dframe_demo_alltestresults[col]/dframe_demo_alltestresults[col+'_baseline']\n",
    "\n",
    "\n",
    "dframe_demo_alltestresults.to_csv( 'output/dframe_demo_alltestresults.csv',index=False)\n",
    "dframe_medication_with_outcomes=dframe_medications.merge(dframe_target,how='right',on='id')\n",
    "dframe_demo_with_medication=dframe_demographics.merge(dframe_medication_with_outcomes,how='left',on='id').merge(dframe_demo_alltestresults[dframe_demo_alltestresults['time']==0][['id','CKD(t=0)']],on='id',how='left')\n",
    "dframe_demo_with_medication.to_csv( 'output/dframe_demo_with_medication.csv',index=False)\n",
    "print(len(dframe_demo_with_medication.id.unique()))\n",
    "dframe_demo_with_medication.head(10)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "609235f3",
   "metadata": {},
   "source": [
    "### 2. Data Discovery & Insights"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "b6651a6f",
   "metadata": {},
   "source": [
    "#### <font color='grey'>Progression Rate by Demographics</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "7fb032b9",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAYUAAAEGCAYAAACKB4k+AAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAASvElEQVR4nO3de9RldV3H8ffHwbyBCjKwJkSH1RovkDrqgApqEF6QMrS8wDIlRekCmnlbWK00c8oytVVKNQoBRRCpIF6WYCMKIgQziNwUnQR1hGAUUjBCGL/9sffze47PnLkxnAvzvF9rPeuc89t7n/M9+znnfPb+7bN/J1WFJEkA95t0AZKk6WEoSJIaQ0GS1BgKkqTGUJAkNTtMuoBtseuuu9bixYsnXYYk3aesXr36+1W1cNi0+3QoLF68mFWrVk26DEm6T0ny7Y1Ns/tIktQYCpKkxlCQJDWGgiSpMRQkSY2hIElqDAVJUmMoSJIaQ0GS1Nynz2iWpHvDB9/8yUmXcK879n0vvEfLuacgSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqTEUJEmNoSBJagwFSVJjKEiSGkNBktQYCpKkxlCQJDWGgiSpMRQkSY2hIElqDAVJUmMoSJKakYVCkj2TnJfka0muTvL7ffsuST6X5Jv95c4Dy7w9yZok1yZ5/qhqkyQNN8o9hbuBN1fV44GnA8ck2Rs4DlhZVUuAlf1t+mmHA/sAhwDHJ1kwwvokSXOMLBSq6saquqy/fhvwNWAP4DDg5H62k4EX9dcPA06vqjur6jpgDbDfqOqTJG1oLMcUkiwGngz8J7B7Vd0IXXAAu/Wz7QF8d2CxtX2bJGlMRh4KSXYEPga8sap+tKlZh7TVkPs7OsmqJKvWrVt3b5UpSWLEoZDk/nSBcGpVfbxvvinJon76IuDmvn0tsOfA4o8Ebph7n1W1oqqWVdWyhQsXjq54SZqHRvntowAnAF+rqvcPTDobOLK/fiTwiYH2w5M8IMlewBLgklHVJ0na0A4jvO8DgFcCVya5vG/7Q+A9wBlJjgK+A7wUoKquTnIGcA3dN5eOqar1I6xPkjTHyEKhqr7E8OMEAAdvZJnlwPJR1SRJ2jTPaJYkNYaCJKkxFCRJjaEgSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqTEUJEmNoSBJagwFSVJjKEiSGkNBktQYCpKkxlCQJDWGgiSpMRQkSY2hIElqDAVJUmMoSJIaQ0GS1BgKkqTGUJAkNYaCJKkxFCRJjaEgSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqTEUJEmNoSBJagwFSVIzslBIcmKSm5NcNdD2ziTfS3J5/3fowLS3J1mT5Nokzx9VXZKkjRvlnsJJwCFD2j9QVUv7v88AJNkbOBzYp1/m+CQLRlibJGmIkYVCVZ0P3LKFsx8GnF5Vd1bVdcAaYL9R1SZJGm4SxxSOTXJF3720c9+2B/DdgXnW9m0bSHJ0klVJVq1bt27UtUrSvDLuUPh74BeApcCNwPv69gyZt4bdQVWtqKplVbVs4cKFIylSkuarsYZCVd1UVeur6qfAh5ntIloL7Dkw6yOBG8ZZmyRpzKGQZNHAzRcDM99MOhs4PMkDkuwFLAEuGWdtkiTYYVR3nOQ04EBg1yRrgXcAByZZStc1dD3w2wBVdXWSM4BrgLuBY6pq/ahqkyQNN7JQqKojhjSfsIn5lwPLR1WPJGnzPKNZktQYCpKkxlCQJDWGgiSpMRQkSY2hIElqDAVJUmMoSJIaQ0GS1BgKkqTGUJAkNYaCJKkxFCRJjaEgSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqTEUJEmNoSBJanbY1MQkv76p6VX18Xu3HEnSJG0yFIAX9pe7AfsDn+9vHwR8ATAUJGk7sslQqKpXAyT5FLB3Vd3Y314EfGj05UmSxmlLjyksngmE3k3AY0ZQjyRpgjbXfTTjC0nOAU4DCjgcOG9kVUmSJmKLQqGqju0POj+rb1pRVWeOrixJ0iRs6Z7CzDeNPLAsSduxzX0l9UtV9cwkt9F1G7VJQFXVQ0danSRprDb37aNn9pc7jaccSdIkeUazJKkxFCRJjaEgSWoMBUlSYyhIkhpDQZLUjCwUkpyY5OYkVw207ZLkc0m+2V/uPDDt7UnWJLk2yfNHVZckaeNGuadwEnDInLbjgJVVtQRY2d8myd504ynt0y9zfJIFI6xNkjTEyEKhqs4HbpnTfBhwcn/9ZOBFA+2nV9WdVXUdsAbYb1S1SZKGG/cxhd1nhuDuL3fr2/cAvjsw39q+bQNJjk6yKsmqdevWjbRYSZpvpuVAc4a01ZA2qmpFVS2rqmULFy4ccVmSNL+MOxRu6n+1bebX227u29cCew7M90jghjHXJknz3rhD4WzgyP76kcAnBtoPT/KAJHsBS4BLxlybJM17W/x7ClsryWnAgcCuSdYC7wDeA5yR5CjgO8BLAarq6iRnANcAdwPHVNX6UdUmSRpuZKFQVUdsZNLBG5l/ObB8VPVIkjZvWg40S5KmgKEgSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqTEUJEmNoSBJagwFSVJjKEiSGkNBktQYCpKkxlCQJDWGgiSpMRQkSY2hIElqDAVJUmMoSJIaQ0GS1BgKkqTGUJAkNYaCJKkxFCRJjaEgSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqTEUJEmNoSBJagwFSVJjKEiSGkNBktTsMIkHTXI9cBuwHri7qpYl2QX4N2AxcD3wsqq6dRL1SdJ8Nck9hYOqamlVLetvHwesrKolwMr+tiRpjKap++gw4OT++snAiyZXiiTNT5MKhQLOTbI6ydF92+5VdSNAf7nbsAWTHJ1kVZJV69atG1O5kjQ/TOSYAnBAVd2QZDfgc0m+vqULVtUKYAXAsmXLalQFStJ8NJE9haq6ob+8GTgT2A+4KckigP7y5knUJknz2dhDIclDkuw0cx14HnAVcDZwZD/bkcAnxl2bJM13k+g+2h04M8nM4/9rVX02yaXAGUmOAr4DvHQCtUnSvDb2UKiqbwFPGtL+A+DgcdcjSZo1TV9JlSRNmKEgSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqTEUJEmNoSBJagwFSVJjKEiSGkNBktQYCpKkxlCQJDWGgiSpmcQvr0maAl989i9NuoR73S+d/8VJl3Cf556CJKlxT2Ee+M67njDpEkbiUX9y5aRLkLY77ilIkhpDQZLUGAqSpMZQkCQ1hoIkqdluv3301LeeMukSRmL1e1816RIkbcfcU5AkNYaCJKkxFCRJjaEgSWoMBUlSYyhIkhpDQZLUGAqSpGa7PXlNGuaAvztg0iWMxIWvv3DSJWg74Z6CJKkxFCRJjaEgSWqmLhSSHJLk2iRrkhw36XokaT6ZqlBIsgD4EPACYG/giCR7T7YqSZo/pioUgP2ANVX1rar6CXA6cNiEa5KkeSNVNekamiQvAQ6pqtf2t18JPK2qjh2Y52jg6P7mY4Frx17ohnYFvj/pIqaE62KW62KW62LWNKyLR1fVwmETpu08hQxp+5nUqqoVwIrxlLNlkqyqqmWTrmMauC5muS5muS5mTfu6mLbuo7XAngO3HwncMKFaJGnembZQuBRYkmSvJD8HHA6cPeGaJGnemKruo6q6O8mxwDnAAuDEqrp6wmVtianqzpow18Us18Us18WsqV4XU3WgWZI0WdPWfSRJmiBDQZLUGApDJHlxkkryuM3M95kkDx9TWWOXZH2Sy5N8NcllSfbv2xcnueoe3ucXkkzd1/GS3D7n9m8l+WB//XeSvGpMdbwryXPG8VhDHnuD/2uSdyZ5yyaWaetpPknygSRvHLh9TpKPDNx+X5I3JfnURpb/yMxoDUn+cOQFbwVDYbgjgC/Rfftpo6rq0Kr6n7FUNBl3VNXSqnoS8HbgLyZd0CRU1T9U1Sljeqw/qar/GMdjaZt8GZjZSLof3Qlp+wxM3x+4/8YWrqrXVtU1/U1DYZol2RE4ADiKPhSSLEpyfr/VfFWSZ/Xt1yfZtb9+VpLVSa7uz7qeub/bkyzvt7YvTrL7BJ7WveGhwK1zG/utywv6PYm2N9FPe1uSK/vn/p45y90vyclJ3j2G2rfJ4NZykjckuSbJFUlOH5j+z0k+n+SbSV7Xt++YZGW/Xq5McljfvjjJ15J8uH+9nJvkQf20k/oz+0myb5Iv9+vvkiQ7TWYNtD28v+zr+MbMe2DOPL+S5KIku/bP42/7+r818JyS5L39++jKJC/v249P8mv99TOTnNhfPyrJuze1zibkQvpQoAuDq4Dbkuyc5AHA44GvADsm+WiSryc5NUlgdo+5f188qP9sObWf9pv9er48yT+mGxNubKbqK6lT4kXAZ6vqG0luSfIU4CDgnKpa3v+DHjxkuddU1S39C/XSJB+rqh8ADwEurqo/SvJXwOuAqf8g7D0oyeXAA4FFwC8Pmedm4LlV9X9JlgCnAcuSvIBuXT6tqv43yS4Dy+wAnApcVVXLR/kEtsLMc52xC8PPkTkO2Kuq7szPdh0+EXg63f/7K0k+TbduXlxVP+o3Hi5OMnOfS4Ajqup1Sc4AfgP4l5k7S3eezr8BL6+qS5M8FLjj3nii22CHqtovyaHAO4DWzZXkxcCbgEOr6tb+s28R8EzgcXTr8qPArwNLgSfRbV1fmuR84HzgWf18e/TL0i9/en99k+tsnKrqhiR3J3kUXThcRFf3M4AfAlcAPwGeTBcaN9AFyQF0vRAz93NckmOrailAkscDLwcOqKq7khwPvAIYy54qGArDHAH8TX/99P72J4ETk9wfOKuqLh+y3Bv6NwZ0Z2UvAX5A98KY6VdcDTx3NGWPxB0DL9ZnAKck+cU589wf+GCSpcB64DF9+3OAf6qq/wWoqlsGlvlH4IwpCgQYeK7Q9ZUDw459XAGcmuQs4KyB9k9U1R3AHUnOoxvc8dPAnyd5NvBTug+NmT3F6wZeR6uBxXMe57HAjVV1KUBV/egePq+tsbHvp8+0f7y/nFvvQXTr6nlz6jyrqn4KXJPZPeRnAqdV1XrgpiRfBPYFLgDemK6f/Rpg5ySL6D5k3wA8gs2vs3Gb2VvYH3g/3f93f7pQ+HI/zyVVtRag3+hYzEAoDHEw8FS6sAR4EN3GxdjYfTQgySPotoY/kuR64K10qX0B8Gzge8A/Z85BxyQH0n0IPqPvf/8K3dY1wF01ezLIeu6jQVxVF9Ft2c0dROsPgJvotvyWAT/Xt4eNf8h8GTgoyQM3Mn2a/Qrd8O5PBVYnmfl/zn2uRbeFtxB4ah84NzH7urhzYN5hr4tNrb9R+QGw85y2XZgdvG2m5rn1fgvYidkNAubMD7Pjmg0b34yq+l7/2IfQ7TVcALwMuL2qbhtyf9PwXpo5rvAEuu6ji+lCbH+6wICtrznAyf2xvKVV9diqeue9WvVmGAo/6yXAKVX16KpaXFV7AtfRBcLNVfVh4ATgKXOWexhwa99N8ji6boTtSv+8FtB9cAx6GN0W7U+BV/bzAJwLvCbJg/vlB7uPTgA+A/z7wIfq1Et3QHHPqjoPeBvwcGDHfvJhSR7Yb1gcSDdky8PoXjd3JTkIePRWPNzXgZ9Psm//2DuNel1V1e3AjUkO7h9zF7oP6U1t2QJ8m65b6JQk+2xm3vOBlydZkGQh3Xvrkn7aRcAbmQ2Ft/SX0+pC4FeBW6pqfb83/HC6YLhoK+7nrr4XAmAl8JIku0H3P0iyNa+bbXafeUOOyRHAe+a0fQw4CfhxkruA24G5X0/8LPA7Sa6gG8r74hHXOS6D/ewBjqyq9f1u7YzjgY8leSlwHvBjgKr6bN+ltCrJT+hCoH3Loqren+RhdHter+hDZdotAP6lrzvAB6rqf/r1cQldd9GjgD/r+5xPBT6ZZBVwOd0H/Rapqp/0B2H/rj9OdQfd3ujtm15ym70K+FCS9/W3/7Sq/mvO/3wDVXVtklfQBf0LNzHrmXQfml+l2xN6W1X9dz/tArouqDVJvk23lzLNoXAl3d7zv85p27Gqvr+5dTZgBXBFksuq6hVJ/hg4t98IuQs4hi54x8JhLqRtlOSddN0cfz3pWqRtZfeRJKlxT0GS1LinIElqDAVJUmMoSJIaQ0GS1BgK0jZIx/eRthu+mKWtNDBi5/HAZcAJSVb1o3f+6cB8G4xy2p/J+94kl6YbafW3J/dMpA15RrN0zzwWeHVV/V6SXfoRchcAK5M8ke7s5WGjnB4F/LCq9k03xPKFSc6tqusm9kykAYaCdM98u6pmhjN5Wbrf0NiBbsjnvemGcNhglNMkzwOemP73BejGR1pCN8aWNHGGgnTP/BggyV50A7ft2/+OwEl0I6FubJTTAK+vqnPGVai0NTymIG2bh9IFxA/73wx4Qd++sVFOzwF+d2ZUzCSPSfKQCdQtDeWegrQNquqrSb4CXE33uwIX9u0bG+X0I3Q/tHJZumE019H9Qp00FRz7SJLU2H0kSWoMBUlSYyhIkhpDQZLUGAqSpMZQkCQ1hoIkqfl/fvTgn1loEYoAAAAASUVORK5CYII=\n",
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
     "data": {
      "text/plain": [
       "<matplotlib.colors.ListedColormap at 0x2288f8412e0>"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dframe_temp=dframe_demographics.merge(dframe_target,how='inner',on='id')\n",
    "\n",
    "\n",
    "\n",
    "my_internal_func.my_hist_by_category(dframe_demo_with_medication,target='id')\n",
    "\n",
    "\n",
    "dframe_progression_rate=pd.DataFrame(dframe_temp.groupby('Age Group',as_index=False).agg({'id':pd.Series.nunique,'target': sum}))\n",
    "dframe_progression_rate.columns=['Age Group','Patient Count','Progress Count']\n",
    "dframe_progression_rate['Progression Rate']=100*dframe_progression_rate['Progress Count']/sum(dframe_progression_rate['Progress Count'])\n",
    "dframe_progression_rate=dframe_progression_rate.sort_values('Progression Rate',ascending=False)\n",
    "sns.color_palette(\"flare\", as_cmap=True)\n",
    "\n",
    "# plt.figure(figsize=(10,6))\n",
    "# ax=sns.barplot('Progression Rate','Age Group',data=dframe_progression_rate)\n",
    "# ax.bar_label(ax.containers[0])\n",
    "# ax.bar_label(ax.containers[0], fmt='%.0f%%')\n",
    "# plt.xlabel(\"Rate of Progression\", size=15)\n",
    "# plt.ylabel(\"Age\", size=15)\n",
    "# plt.title(\"Age wise Progression of Kidney Disease\", size=18)\n",
    "# plt.tight_layout()\n",
    "# plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "a5eaa085",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "<seaborn.axisgrid.FacetGrid at 0x22891746ca0>"
      ]
     },
     "execution_count": 9,
     "metadata": {},
     "output_type": "execute_result"
    },
    {
     "data": {
      "text/plain": [
       "<Figure size 1440x1440 with 0 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAABDAAAALICAYAAACJhQBYAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAABk7UlEQVR4nOzdeZglZX33//dHBgMMBCISQBGGGBVZB2mJKCIYNe4rikqA0SgxRkl8IsZExSHEFWMSJWpGgwOKiiAQonlERTZBlh6YhVXzIPkpGhU0ygAiy/f3x7kbDm3v0zOneub9uq6+Tp2qu+76Vp3TM9Wfc1edVBWSJEmSJEld9pBBFyBJkiRJkjQZAwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4ChDUaSrZMsbz//k+TmvucPnWGfByZ58mzXuja3k+SmJBeNmrc8ydXT7GdpkoPb9KeS7DqDWhYmeW7f8xcmeft0+xmn73tH9ivJaUk2m8a6j0hy+jS3d36SoTHmH5jkF0muSnJDkguTPL9v+RuSHD5J3/cf6ynWsiDJq6dT/zj9nNmO4X+1fRj5fZnV93ySzZJ8Jcn1Sa5J8v7Z7F+S1ibPL+7vy/OLyded7fOLL4+aN63zhTWR5I/63uer2znO8iQnz0LfQ0k+Mht1av0zb9AFSOtKVd0KLARIshhYXVUfGlmeZF5V3TPNbg8EVgOXzE6V62w7WyR5VFV9P8nj17SzqnrdDFddCAwB/9n6ORs4e03rae6sqoUASU4B3gB8eLKV2vvgh8BsngBcVFXPb/0vBM5KcmdVnVtVn5jF7YxYALwa+NyadFJVL4HeSRLw1pF9WEs+VFXntZP9c5M8p6r+71rcniTNCs8vHsTzi3GspfOLdar/vVxV5wDntPnn0ztPGJ6N7bR+ZqUvrX8cgaENWkuqP5zkPOADSR6d5KtJliW5KMkurd0LklzWPkX/RpJtkyyg95/WW1ri/NTW38eTnJfkxiRPS3JikuuSLO3b7rOSfDvJlS2937zNvynJsW3+qiS7jLWdWdj1LwKHtOlXAZ/vq22jJMcnuSLJyiR/2uYnyQlJrk3yFeB3+9a5/9OBJM9u9a9Icm6bt2+SS9rxuyTJ49ofqn8HHNL265Aki5Kc0NbZKcm5rYZzk+zY95p9pPVz4xQ/abgI+P0k89vrcUWr5UWtz0XtdfgP4GvpjWC4ui3bJMmn2+txVZKD2vxNk3yh1XcqsOlUDnxVLW/7/abWz+Ikb23Tr2+1rUjypTz4U51ntPfkd9JGcIz3WgHvB57ajutbJnhNt09vRMjIJ0mTvrdaDQv7nl+cZM+2H59J8s0k303y+r42R/dt+9gxjskdVXVem/41cCWww1SOpyR1UTy/AM8v1un5Rb8kf5jkzL7nz0xyRpteneQf2rE8N8k2bf5479EHvZensO0/TnJ5O/b/mmSjvu2+p71+lybZts1/eXrnICuSXNjm3T+6JMnDkpzVjselSfZs8xe3Y35+e72Omu5x0hxVVf74s8H9AIuBtwJLgS8DG7X55wKPadN/AHyzTf8OkDb9OuAf+vvp63cp8AUgwIuAXwJ70AsLl9H7RODhwIXA/LbOXwPHtOmbgDe36TcCnxprO6P25SBg+Rg/l4zT/ibgsSPLgauAXYGr2/MjgXe26d+il4DvDLwU+DqwEfAI4H+Bg1u78+l90rEN8H1g5zb/Ye3xt4F5bfoZwJfa9CLghL7a7n8O/AdwRJt+LXBW3zE+rR3TXYH/Gmc/V7fHecC/A38GvBf44zZ/K+A7wPy23R/01bug73j8FfDpNr0L8P8BmwD/Bzixzd8TuAcYGqOOA4Evj5q3ELhu9GsLbN3X5u954L2wFPhq2+fHtFo3meC1etA2J2j3V8A72vyNgC3GOZb39wccAfxTm34sMNy3HyvonWg9vL0PHgE8C1hC73fiIfR+3w6Y4HdzK+BG4PcG/e+EP/744890f/D8wvOLdXt+8YtRr83P6I3wCHA9sE1r+zngBW26gEPb9DF9x2W89+hS+t7L4xyTkdfp8e34btzmfww4vG+7IzV8kAfeC6uAR44cu759Gznv+Cjw7jb9dGB533v3EnrvpYcDt45s15/1+8dLSCQ4rarubZ9SPBk4LcnIst9qjzsApybZHngo8L0J+vuPqqokq4AfV9UqgCTX0PuPawd6/zFe3LbzUODbfeuf0R6X0ftPfULV++R64WTtRvkZ8PMkrwSuA+7oW/YsYM++Tx62pPdH8wHA56vqXuCHSb45Rr9PAi6squ+12n7W18dJSR5D7z+wjadQ4348sP+fofef3Yizquo+4NqRBH8MmyZZ3qYvAv6N3n90L0wb8UDvRGHHNv31vnr77U/vP0+q6vok/03vBO0A4CNt/sokK6ewTyMyzvzdk/w9vZOfzWlDM5svtn3+bpIb6Z3sjPda/XpUv+O1uwI4McnG9I7p8inUfhrwriRH0zvxW9q37N+r6k7gzvZJzb70jt+z6J3I0vbrMfROsh8kyTx6n9Z9pKpunEItktRlnl94fgFr9/zi/ktUoTdaoq1XST4D/HGST7d9Hrnf1n3AqW36s8AZk7xHob2XJ6hjxB8C+wBXtH42BX7Slv2aXhACvffgM9v0xcDSJF/kgfdov/2Bl7X9+mZ695zZsi37SlXdBdyV5CfAtvQCI63HDDAkuL09PgT432rXNY7yUeDDVXV2evcDWDxBf3e1x/v6pkeezwPupfef2asmWf9epvA72oYc/uMYi+6oqoluzHUq8C/0Ph14UJf0PqU550EzezfDqsnKGafNccB5VfWS9Iasnj9JP2Pp77f/uI4XBtw5+rVM73/Tl1XVDaPm/wEPvA9GG6//0TWN9PUS4N3t6XjX7u5N78RutKXAi6tqRZJF9D6BGG9bxfiv1YGj2o7ZrrU9AHge8Jkkx1fVhDffqqo7knyd3ieAr6D3ictkNb6vqv51on6bJcB3q+qfptBWkrrO84tRXeL5xYMWT7Gmkb6mcn7R79P0RkP8il4AMd59WIqJ36Mw/j78RpnASVX1N2Msu7uqRvbr/vdgVb2hHafnAcvTd5lqX59j1QwPfr2m9L7W3Oc9MKSmqn4JfC/Jy+H+azL3aou3BG5u00f0rXYbsMU0N3Up8JQkv9+2s1mSx06yzrjbqarzqmrhGD+T3VX8THqfOoz+o/Yc4M/ap/IkeWyS+fQ+MX9letewbk9vaOlo3waelmTntu7D2vz+47doKvtF79OMV7bpQ4FvTbI/U3EO8OZ2okGSvaewzoVt+7TXaUfghlHzd6c3zJOqOrPvNfiNG1C1azffRe/kbrQtgB+1Y3/oqGUvT/KQJI8Gfq/VMN5rNfq4jtkuyU7AT6rqk/Q+QXrCFI4HwKfofTp0xahPlV6U3jW9W9MLX65o235tHrgO+5FJfnd0h23kyZbAX06xBkmaEzy/uJ/nFw82q+cXo1XvhqE/BN7Jg0dLPoQHbiT6auBbk7xHp+Nc4OCR/+fTu3/FThOtkOTRVXVZVR0D3AI8alST/uNxIHBLq1cbKAMM6cEOBf4kyQrgGnqfMkPvE5HT0vt6sFv62v8H8JJM4+ZXVfVTev/Jfr4NC7yU3uUAE5n2dqZQx21V9YHq3TSx36eAa4Er07vR1L/SS7TPBL5L71rFjwMXjNHnT+ld43pGO4YjQxQ/CLwvycX0rnEdcR6wa9uvQx7cG0cBr2nH6DDgL2a+t/c7jt7w0pVt346bwjofAzZKb8juqcCiNlzx48Dmrb63AZdP0MdT075GlV5wcVRVnTtGu3cBl9G7Fvj6UctuoHfM/y/whqr6FeO/ViuBe9K7IdZbJmh3IL1PO66iNzzzn6dwPKiqZfSuv/70qEWXA1+h954+rqp+WFVfo3ft7bfbMTydUSeVSXYA3kFv6POV7f0w0zvPS1IXeX7h+cVos3F+MZlTgO9X1bV9824HdkuyjN49Jf6uzR/vPTplbTvvpHfD0pX0zme2n2S149O7kenV9MKKFaOWLwaGWn/v58FBnzZAIzcNkiRpSpI8gt4w3V3atcJkjK8OlCRJg5PeN69cVVX/1jdvdVVtPsCypDXiCAxJ0pQlOZzeKJF3jIQXkiSpW9oIiz3p3ahTWm84AkOSJEmSJHWeIzAkSZIkSVLnGWBIkiRJkqTO87tyZ+jZz352ffWrXx10GZIkdU0GXcBc4bmEJEnjGvN8whEYM3TLLbdM3kiSJGkcnktIkjQ9BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJ05Bk9STLFyS5el3VI0nShsIAQ5IkdVqSjdZw/XmzVYskSRocAwxJkjQwbbTC9UlOSrIyyelJNktyU5JjknwLeHmSVyVZleTqJB/oW391kn9IcmWSc5Ns0+afn+S9SS4A/iLJC5JcluSqJN9Ism1rtzjJia39jUmO6uv7/7TtXZ3kL8eoPUmOb8tXJTlkrR8wSZI2YAYYkiRp0B4HLKmqPYFfAm9s839VVfsDFwIfAJ4OLASemOTFrc184MqqegJwAfDuvn63qqqnVdU/AN8CnlRVewNfAN7W124X4I+AfYF3J9k4yT7Aa4A/AJ4EvD7J3qPqfmmrZy/gGcDxSbafaEeTHJlkOMnwT3/608mPjCRJup8BhiRJGrTvV9XFbfqzwP5t+tT2+ETg/Kr6aVXdA5wCHNCW3dfXrn/d/vUBdgDOSbIKOBrYrW/ZV6rqrqq6BfgJsG3r58yqur2qVgNnAE8dVff+wOer6t6q+jG9AOWJE+1oVS2pqqGqGtpmm20maipJkkYxwJAkSYNW4zy/vT1mhn3d3jf9UeCEqtoD+FNgk75ld/VN3wvMm+I2p1OXJElaQwYYkiRp0HZMsl+bfhW9yz36XQY8LcnD2w09X0VvtAP0zmUObtOvHmPdEVsCN7fpI6ZQ04XAi9v9OOYDLwEuGqPNIUk2avfeOAC4fAp9S5KkGTDAkCRJg3YdcESSlcDDgI/3L6yqHwF/A5wHrKB3z4t/b4tvB3ZLsozePTL+bpxtLAZOS3IRcMtkBVXVlcBSeoHEZcCnquqqUc3OBFa2mr4JvK2q/meyviVJ0sykavSoTU3F0NBQDQ8PD7oMSZK6ZlqXVSRZAHy5qnaf0caS1VW1+UzWHTTPJSRJGteY5xOOwJAkSZIkSZ03b9AFSJKkDVdV3QTMaPRFW39Ojr6QJEnT5wgMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZpzkqyeZPmCJFdPs89FSR4xhXZLkxw8SZvzkwxNZ/uSJGliBhiSJEk9i4BJAwxJkjQYBhiSJGnOSrJ5knOTXJlkVZIX9S2el+SkJCuTnJ5ks7bOPkkuSLIsyTlJtm8jKoaAU5IsT7JpkmOSXJHk6iRLkmSM7f9hkqvatk9M8lvraNclSdrgGGBIkqS57FfAS6rqCcBBwD/0BQ2PA5ZU1Z7AL4E3JtkY+ChwcFXtA5wIvKeqTgeGgUOramFV3QmcUFVPrKrdgU2B5/dvOMkmwFLgkKraA5gH/Nla3l9JkjZYBhiSJGkuC/DeJCuBbwCPBLZty75fVRe36c8C+9MLNXYHvp5kOfBOYIdx+j4oyWVJVgFPB3YbtfxxwPeq6jvt+UnAARMWmxyZZDjJ8E9/+tOp7qMkSaL3SYEkSdJcdSiwDbBPVd2d5CZgk7asRrUteoHHNVW130SdttEVHwOGqur7SRb39Xt/s+kWW1VLgCUAQ0NDo+uTJEkTcASGJEmay7YEftLCi4OAnfqW7ZhkJKh4FfAt4AZgm5H5STZOMjKy4jZgizY9ElbckmRzYKxvHbkeWJDk99vzw4ALZmOnJEnSbzLAkCRJc9kpwFCSYXqjMa7vW3YdcES7vORhwMer6tf0wogPJFkBLAee3NovBT7RLi25C/gksAo4C7hi9Iar6lfAa4DT2mUm9wGfmN3dkyRJI1Ll6MWZGBoaquHh4UGXIUlS10z7sooNlecSkiSNa8zzCUdgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdd60A4wki5O8dW0UM8a2zk8y1Kb/M8lWbfqoJNclOSXJoiQnzKDvFyZ5+yyXLEmSJEmS1oJ5gy5gqqrquX1P3wg8p6q+l2TRDPs7Gzh7NmqTJEmSJElr15RGYCR5R5IbknwDeFyb9/okVyRZkeRLSTZr85cm+USSi5J8J8nz2/wFbd6V7efJbf5DknwsyTVJvtxGWhw8Rg03JXl4kk8AvwecneQto9rslOTcJCvb445t/vK+nzuTPK1/5Ear+SNJLkly41jblyRJkiRJgzNpgJFkH+CVwN7AS4EntkVnVNUTq2ov4DrgT/pWWwA8DXge8IkkmwA/AZ5ZVU8ADgE+0tq+tLXfA3gdsN9E9VTVG4AfAgdV1T+OWnwCcHJV7QmcMrKNqlpYVQuBdwHDwCVjdL09sD/wfOD9E9UgSZIkSZLWralcQvJU4MyqugMgychlF7sn+XtgK2Bz4Jy+db5YVfcB301yI7AL8D3ghCQLgXuBx7a2+wOntfb/k+S8Ndif/egFIgCfAT44siDJY4DjgadX1d1JRq97Vqvh2iTbjtV5kiOBIwF23HHHNShTkiRJkiRNx1Rv4lljzFsKvKmq9gCOBTaZoH0BbwF+DOwFDAEPbct+I0mYRQWQZD7wReD1VfXDcdre1Tc9Zk1VtaSqhqpqaJtttpndSiVJkiRJ0rimMgLjQmBpkve39i8A/hXYAvhRko2BQ4Gb+9Z5eZKTgJ3p3a/iBmBL4AdVdV+SI4CNWttvAUe09tsABwKfm+H+XELvcpfPtJq+1eZ/Gvh0VV00w34lSZJm1XU/uJV9jj55xusvO/7wWaxGkqTumzTAqKork5wKLAf+GxgJAd4FXNbmraIXaIy4AbgA2BZ4Q1X9KsnHgC8leTlwHnB7a/sl4A+Bq4HvtD5/McP9OQo4McnRwE+B1yTZCTgYeGyS17Z2r5th/5IkSZIkaQBSNdbVIWvQYbIU+HJVnT6NdTavqtVJtgYuB55SVf8zq4XNsqGhoRoeHh50GZIkdc3avDR0vTJ/u51rl8OOnfH6jsCQJK3HxjyfmMolJOvCl5NsRe++GMd1PbyQJEmSJEnr1qwHGFW1aAbrHDjbdUiSJEmSpPXHVL+FRJIkSZIkaWAMMCRJkiRJUucZYEiSpM5JsnrQNQAk+dtB1yBJknoMMCRJkkZJz0MAAwxJkjrCAEOSJA1Ukv+T5Or285ejlh2Y5IIkX0zynSTvT3JoksuTrEry6NZumyRfSnJF+3lKm/+0JMvbz1VJtkiyeZJzk1zZ+nhRa7sgyXVJPgZcCfwbsGlb95TW5qwky5Jck+TIvjpXJ3lPkhVJLk2y7bo5epIkbTi68jWqkiRpA5RkH+A1wB/Q+873y5JcMKrZXsDjgZ8BNwKfqqp9k/wF8GbgL4F/Bv6xqr6VZEfgnLbOW4E/r6qLk2wO/Kr1+ZKq+mWShwOXJjm7zX8c8JqqemOr7+VVtbCvltdW1c+SbApckeRLVXUrMB+4tKrekeSDwOuBv5+doyRJksAAQ5IkDdb+wJlVdTtAkjOAp45qc0VV/agt/3/A19r8VcBBbfoZwK5JRtb57SRbABcDH24jKM6oqh8k2Rh4b5IDgPuARwIjIyb+u6ounaDeo5K8pE0/CngMcCvwa+DLbf4y4JljrdxGbRwJ8NAttp5gM5IkaTQDDEmSNEiZvAl39U3f1/f8Ph44l3kIsF9V3Tlq3fcn+QrwXHojLZ4BPAnYBtinqu5OchOwSWt/+7iFJgfSC0r2q6o7kpzft97dVVVt+l7GOceqqiXAEoD52+1cY7WRJElj8x4YkiRpkC4EXpxksyTzgZcAF82gn68Bbxp5kmRhe3x0Va2qqg8Aw8AuwJbAT1p4cRCw0wT93t1GbNDW+3kLL3ahF4RIkqR1xBEYkiRpYKrqyiRLgcvbrE9V1VV9l4JM1VHAvyRZSe/85kLgDcBftpDiXuBa4P8CWwD/kWQYWA5cP0G/S4CVSa4EXgu8oW3jBmCiS00kSdIsywOjHTUdQ0NDNTw8POgyJEnqmmknDxuq+dvtXLscduyM1192/OGzWI0kSZ0y5vmEl5BIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPL9GVZIkaQAev8PWDPtNIpIkTZkjMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPm3hKkiQNwHU/uJV9jj55xusv8wagkqQNjCMwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmS1GlJFiS5eoz5n0qy6wDqWZzkrZO0WZrk4HVVkyRJG4J5gy5AkiRpJqrqdWur7yTzquqetdW/JEmaPkdgSJKkuWBekpOSrExyepLNkpyfZCjJRm3Ew9VJViV5C0CShUkubeucmeR32vzzk/xTkkvaOvu2+YuTLEnyNeDkJDslObetf26SHUcXNd42JEnS7DPAkCRJc8HjgCVVtSfwS+CNfcsWAo+sqt2rag/g023+ycBft3VWAe/uW2d+VT259XNi3/x9gBdV1auBE4CT2/qnAB8Zo66JtiFJkmaRAYYkSZoLvl9VF7fpzwL79y27Efi9JB9N8mzgl0m2BLaqqgtam5OAA/rW+TxAVV0I/HaSrdr8s6vqzja9H/C5Nv2ZUdtkCtv4DUmOTDKcZPieO26bdKclSdIDDDAkSdJcUOM9r6qfA3sB5wN/DnxqDfq7fRrrTFtVLamqoaoamrfZFmvanSRJGxQDDEmSNBfsmGS/Nv0q4FsjC5I8HHhIVX0JeBfwhKr6BfDzJE9tzQ4DLujr75C27v7AL1r70S4BXtmmD+3fJsAUtiFJkmaR30IiSZLmguuAI5L8K/Bd4OPAC9qyRwKfTjLywczftMcjgE8k2YzeZSav6evv50kuAX4beO042zwKODHJ0cBPR60/YqJtSJKkWWSAIUmSOq2qbgJ2HWPRgX3TTxhjveXAk8bp9ktV9Tf9M6pq8RjbffoY/S7umx5zG1W1aJztSpKkGfISEkmSJEmS1HmOwJAkSRuUqjpw0DVIkqTpcwSGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp87wHhiRJ0gA8foetGT7+8EGXIUnSnOEIDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM7zJp6SJEkDcN0PbmWfo08eaA3LvImoJGkOcQSGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmS1ltJliY5eIz5Q0k+MsM+Fyd560y2K0mSZm7eoAuYq6b63e1+v7okSd1TVcPA8KDrkCRJU+cIDEmSNGckWZDkuiSfTHJNkq8l2TTJo5N8NcmyJBcl2aVvtWe0ed9J8vzWz4FJvpzkIUluSrJV3zb+K8m2SXZKcm6Sle1xxzHqWZjk0tbmzCS/s/aPgiRJGyYDDEmSNNc8BviXqtoN+F/gZcAS4M1VtQ/wVuBjfe0XAE8Dngd8IskmIwuq6j7g34GXACT5A+CmqvoxcAJwclXtCZwCjHXJycnAX7c2q4B3z95uSpKkfgYYkiRprvleVS1v08voBRRPBk5Lshz4V2D7vvZfrKr7quq7wI1A/+gMgFOBQ9r0K9tzgP2Az7XpzwD796+UZEtgq6q6oM06CThgosKTHJlkOMnwPXfcNsluSpKkft4DQ5IkzTV39U3fC2wL/G9VLRynfU3y/NvA7yfZBngx8PdT7GfaqmoJvdEizN9u5zXuT5KkDYkjMCRJ0lz3S+B7SV4OkJ69+pa/vN3r4tHA7wE39K9cVQWcCXwYuK6qbm2LLqE3IgPgUOBbo9b7BfDzJE9tsw4DLkCSJK0VjsCQJEnrg0OBjyd5J7Ax8AVgRVt2A71gYVvgDVX1qySj1z8VuAJY1DfvKODEJEcDPwVeM8Z2j6B3X43N6F2eMlYbSZI0CwwwJEnSnFFVNwG79z3/UN/iZ4/RftE4/ZwPnN/3fBjIqDY3AU8fY93FfdPLgSdNdbuSJGnmvIREkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5/ktJJIkSQPw+B22Zvj4wwddhiRJc4YjMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPm3hKkiQNwHU/uJV9jj550GWskWXehFSStA45AkOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIGIslWSd44hXarp9nvG5IcPvPK7u9nQZJXr2k/kiRpdhhgSJKkQdkKmDTAmK6q+kRVnbwmfSSZBywADDAkSeqIeYMuQJIkbbDeDzw6yXLg68BPgFcAvwWcWVXv7m+c5EDgWODHwELgDGAV8BfApsCLq+r/JVkMrK6qDyU5CngDcA9wbVW9Msm+wD+1de4EXlNVNyRZBDwP2ASYD2wGPL7VdxJwJvCZtgzgTVV1SatrMXALsDuwDPjjqqrZOEiSJKnHAEOSJA3K24Hdq2phkmcBBwP7AgHOTnJAVV04ap29gMcDPwNuBD5VVfsm+QvgzcBfjrGNnavqriRbtXnXAwdU1T1JngG8F3hZW7YfsGdV/awFE2+tqucDJNkMeGZV/SrJY4DPA0Ntvb2B3YAfAhcDTwG+NXqHkxwJHAnw0C22nvKBkiRJBhiSJKkbntV+rmrPNwceA4wOMK6oqh8BJPl/wNfa/FXAQWP0uxI4JclZwFlt3pbASS2EKGDjvvZfr6qfjVPjxsAJSRYC9wKP7Vt2eVX9oNW1nN7lJ78RYFTVEmAJwPztdnaEhiRJ02CAIUmSuiDA+6rqXydpd1ff9H19z+9j7POa5wEHAC8E3pVkN+A44LyqekmSBcD5fe1vn2Dbb6F3+cpe9O4j9qtx6rp3nFokSdIa8CaekiRpUG4DtmjT5wCvTbI5QJJHJvndNek8yUOAR1XVecDb6N00dHN6IzBubs0WTbE+2no/qqr7gMOAjdakPkmSND0GGJIkaSCq6lbg4iRXA88EPgd8O8kq4HQeHB7MxEbAZ1t/VwH/WFX/C3wQeF+Si5k4hFgJ3JNkRZK3AB8DjkhyKb3LRyYarSFJkmZZvEH2zMzfbufa5bBjJ2237Pg1/hp6SZLmkgy6gLliqucSXeZ5jiRpLRnzfMIRGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ3X2e8oT7IYWF1VH0ryd8CFVfWNJE8FPgHcDbwKOK2qdp9m348APlJVB8923ZIkSVPx+B22ZtibYEqSNGVzYgRGVR1TVd9oTw8FPlRVC4E7Z9jfDw0vJEmSJEmaO6YUYCQ5PMnK9j3on0nygiSXJbkqyTeSbNvaLU5yYpLzk9yY5Ki+Ps5KsizJNUmO7Jv/J0m+09b5ZJITxtj+0iQHJ3kd8ArgmCSnjGqzSZJPJ1nV6jqozf9UkuXt56dJ3p1kQfvOeZIsSnJGkq8m+W6SD87kQEqSJEmSpLVn0ktIkuwGvAN4SlXdkuRhQAFPqqpqocLbgL9qq+wCHARsAdyQ5ONVdTfw2qr6WZJNgSuSfAn4LeBdwBOA24BvAivGq6WqPpVkf+DLVXV6kgV9i/+8tdkjyS7A15I8tqpe1/ZjJ+AcYCm/+Z2yC4G9gbtazR+tqu+PcSyOBI4EeOgWW0926CRJkiRJ0iyZygiMpwOnV9UtAFX1M2AH4Jwkq4Cjgd362n+lqu5q7X8CbNvmH5VkBXAp8CjgMcC+wAVV9bMWcpy2BvuyP/CZVuP1wH8Dj4Xe6IzW95uq6r/HWPfcqvpFVf0KuBbYaawNVNWSqhqqqqF5m22xBqVKkiRJkqTpmMpNPENvxEW/jwIfrqqzkxwILO5bdlff9L3AvNbmGcB+VXVHkvOBTfjNkRBrYqK+PgGc0XcfjdF+o+ZZq0qSJGkM1/3gVvY5+uRBlzGnLfMmqJK0QZnKCIxzgVck2RqgXUKyJXBzW37EFPrYEvh5Cy92AZ7U5l8OPC3J7ySZB7xsWtU/2IX0bvBJkscCO9K7HOTPgS2q6v1r0LckSZIkSRqgSUcaVNU1Sd4DXJDkXuAqeiMuTktyM71LQnaepJuvAm9IshK4oa1DVd2c5L3AZcAP6V2+8YsZ7svHgE+0y1ruARZV1V1J3grcnWR5a/eJVo8kSZIkSZojUjX66pB1XECyeVWtbiMwzgROrKozB1rUFMzfbufa5bBjJ23n0EZJ0gZmNi8PXa9N9VxC4/M8S5LWW2OeT0zpa1TXssVtdMTVwPeAswZajSRJkiRJ6pyB36yyqt466BokSZIkSVK3dWEEhiRJkiRJ0oQMMCRJkiRJUucZYEiSpM5Jcsk48xckuXod1rEgyavX1fYkSdL4DDAkSVLnVNWTR89LstG6rKF9Q9oCwABDkqQOMMCQJEmdk2R1ezwwyXlJPgesaovnJTkpycokpyfZrLXdJ8kFSZYlOSfJ9m3+UUmube2/0Obtm+SSJFe1x8e1+YuSnJbkP4CvAe8HnppkeZK3tBEZFyW5sv08ua/O81s91yc5JYlfKStJ0iwa+LeQSJIkTWJfYPeq+l6SBcDjgD+pqouTnAi8Mck/Ax8FXlRVP01yCPAe4LXA24Gdq+quJFu1Pq8HDqiqe5I8A3gv8LK2bD9gz6r6WZIDgbdW1fMBWljyzKr6VZLHAJ8Hhtp6ewO7AT8ELgaeAnyrf0eSHAkcCfDQLbaetQMkSdKGwABDkiR13eVV9b2+59+vqovb9GeBo4CvArsDX28DHzYCftTarAROSXIWcFabtyVwUgshCti4r/+vV9XPxqllY+CEJAuBe4HHjqrzBwBJltO7/ORBAUZVLQGWAMzfbueaeLclSVI/AwxJktR1t496PvoP/wICXFNV+42x/vOAA4AXAu9KshtwHHBeVb2kjeo4f4Lt9XsL8GNgL3qX4v6qb9ldfdP34nmWJEmzyntgSJKkuWbHJCNBxavojXK4AdhmZH6SjZPsluQhwKOq6jzgbcBWwOb0RmDc3PpYNMG2bgO26Hu+JfCjqroPOIzeSA9JkrQOGGBIkqS55jrgiCQrgYcBH6+qXwMHAx9IsgJYDjyZXsDw2SSrgKuAf6yq/wU+CLwvycVMHEKsBO5JsiLJW4CPtW1fSu/ykYlGa0iSpFmUKi+/nIn52+1cuxx27KTtlh1/+DqoRpKkzvCbN6ZoqucSGp/nWZK03hrzfMIRGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ3n95NLkiQNwON32Jphb0IpSdKUOQJDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp87yJpyRJ0gBc94Nb2efokwddhtbAMm/CKknrlCMwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSWNIcn6SoTb9n0m2atOrJ1lvQZKr10GJkiRtUOYNugBJkqTZlmReVd0zW/1V1XNnqy9JkjQzjsCQJEmdluTwJCuTrEjymSQvSHJZkquSfCPJtq3d4iRLknwNOLm12a2vn/OT7JNkfpITk1zR+nhRW75pki+0bZ0KbNq37k1JHj6qriQ5PsnVSVYlOWTdHBFJkjZMjsCQJEmd1QKIdwBPqapbkjwMKOBJVVVJXge8Dfirtso+wP5VdWeStwCvAN6dZHvgEVW1LMl7gW9W1WvbZSGXJ/kG8KfAHVW1Z5I9gSsnKe+lwEJgL+DhwBVJLpxkf44EjgR46BZbT+9gSJK0gXMEhiRJ6rKnA6dX1S0AVfUzYAfgnCSrgKOB3fran11Vd7bpLwIvb9OvAE5r088C3p5kOXA+sAmwI3AA8Nm2nZXAyklq2x/4fFXdW1U/Bi4AnjjRClW1pKqGqmpo3mZbTNK9JEnqZ4AhSZK6LPRGXPT7KHBCVe1Bb9TEJn3Lbh+ZqKqbgVvbaIpDgC/09fmyqlrYfnasqutGVptmbZIkaR3xEpIZevwOWzN8/OGDLkOSpPXducCZSf6xqm5tl5BsCdzclh8xyfpfoHeJyZZVtarNOwd4c5I3t8tQ9q6qq4ALgUOB85LsDuw5Sd8XAn+a5CTgYfRGcBzNgwMVSZI0SxyBIUmSOquqrgHeA1yQZAXwYWAxcFqSi4BbJunidOCV9C4nGXEcsDGwsn3d6XFt/seBzZOspBd6XD5J32fSu8xkBfBN4G1V9T9T3DVJkjRNqZrOSEmNGBoaquHh4UGXIUlS13hZxRTN327n2uWwYwddhtbAMkfjStLaMub5hCMwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs+vUZUkSRoAv5JdkqTpcQSGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzvNbSCRJkgbguh/cyj5HnzzoMjSHLfNbbCRtYByBIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUuf5Naoz5FefSZLmuvX9KxiT/G1VvXct9LsYWF1VH5qgzVLgy1V1+mxvX5KkDZUjMCRJ0vrqbwddgCRJmj0GGJIkac5LclaSZUmuSXJkkvcDmyZZnuSU1uaPk1ze5v1rko3a/NVJ3pNkRZJLk2zb5u+U5NwkK9vjjmNsd2FbZ2WSM5P8zjrdcUmSNiAGGJIkaX3w2qraBxgCjgKOB+6sqoVVdWiSxwOHAE+pqoXAvcChbd35wKVVtRdwIfD6Nv8E4OSq2hM4BfjIGNs9Gfjr1mYV8O6JimzhynCS4XvuuG0NdleSpA2PAYYkSVofHJVkBXAp8CjgMaOW/yGwD3BFkuXt+e+1Zb8GvtymlwEL2vR+wOfa9GeA/fs7TLIlsFVVXdBmnQQcMFGRVbWkqoaqamjeZltMeeckSZI38ZQkSXNckgOBZwD7VdUdSc4HNhndDDipqv5mjC7urqpq0/cy/vlRjTNfkiStA47AkCRJc92WwM9beLEL8KQ2/+4kG7fpc4GDk/wuQJKHJdlpkn4vAV7Zpg8FvtW/sKp+Afw8yVPbrMOAC5AkSWuFIzAkSdJc91XgDUlWAjfQu4wEYAmwMsmV7T4Y7wS+luQhwN3AnwP/PUG/RwEnJjka+CnwmjHaHAF8IslmwI3jtJEkSbPAAEOSJM1pVXUX8JwxFp0P/HVfu1OBU8dYf/O+6dOB09v0TcDTx2i/uG96OQ+M+Ohvs2iq9UuSpKnxEhJJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTO8yaekiRJA/D4HbZm+PjDB12GJElzhiMwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ7fQiJJkjQA1/3gVvY5+uRBlyENzDK/hUfSNDkCQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSeudJKtnqZ9FSR4xG31JkqQ1Y4AhSZI0hiQbAYsAAwxJkjrAAEOSJHVekg8keWPf88VJ3p3k3CRXJlmV5EVjrLd9kguTLE9ydZKntvkfTzKc5Jokx/a1vynJMUm+BbwKGAJOaetv2pZd0fpakiRtvfNbjZcn+c7IdiRJ0uwxwJAkSXPBF4BD+p6/Avg08JKqegJwEPAPI4FCn1cD51TVQmAvYHmb/46qGgL2BJ6WZM++dX5VVftX1WeBYeDQqlpYVXcCJ1TVE6tqd2BT4Pl9682rqn2BvwTePdZOJDmyBSfD99xx2zQPgSRJGzYDDEmS1HlVdRXwu0kekWQv4OfAj4D3JlkJfAN4JLDtqFWvAF6TZDGwR1WNpAavSHIlcBWwG7Br3zqnTlDKQUkuS7IKeHpbd8QZ7XEZsGCc/VhSVUNVNTRvsy0m3GdJkvRg8wZdgCRJ0hSdDhwMbEdvRMahwDbAPlV1d5KbgE36V6iqC5McADwP+EyS44GLgLcCT6yqnydZOmq928faeJJNgI8BQ1X1/RaK9K93V3u8F8+xJEmadY7AkCRJc8UXgFfSCzFOB7YEftLCi4OAnUavkGSn1uaTwL8BTwB+m15I8Ysk2wLPmWCbtwEjQyVGwopbkmze6pAkSeuInw5IkqQ5oaquSbIFcHNV/SjJKcB/JBmmd2+L68dY7UDg6CR3A6uBw6vqe0muAq4BbgQunmCzS4FPJLkT2A/4JLAKuIne5SmSJGkdMcCQJElzRlXt0Td9C71QYax2m7fHk4CTxli+aJz1Fox6/iXgS32z3tl+Rq934Ki6FoxuI0mS1oyXkEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHVep27imWQh8Iiq+s/2/LeArwAPB95XVafO4rYeAXykqvwKNEmStM49foetGT7+8EGXIUnSnNGpAANYCAwB/9me7w1sXFULp9pBko2q6t7J2lXVD/H72yVJkiRJmhNm/RKSJAuSXJ/kU0muTnJKkmckuTjJd5Psm2R+khOTXJHkqiQvSvJQ4O+AQ5IsT3II8FlgYXv+6CR/2Nqvauv/VtvmTUmOSfIt4OXt+XuTfDvJcJInJDknyf9L8oa+Oq9u04uSnJHkq63GD872cZEkSZIkSTO3tu6B8fvAPwN7ArsArwb2B94K/C3wDuCbVfVE4CDgeGBj4Bjg1Kpa2C4XeR1wURuBcTOwFDikfQf8PODP+rb5q6rav6q+0J5/v6r2Ay5q6x0MPIleSDKWhcAhwB70QpRHjW6Q5MgWiAzfc8dt0z0mkiRJkiRphtZWgPG9qlpVVfcB1wDnVlUBq4AFwLOAtydZDpwPbALsOEmfj2v9fqc9Pwk4oG/56PtjnN0eVwGXVdVtVfVT4FdJthqj/3Or6hdV9SvgWmCn0Q2qaklVDVXV0LzNtpikXEmSJEmSNFvW1j0w7uqbvq/v+X1tm/cCL6uqG/pXSvIHE/SZSbZ5+zg19G+/v4aJar53nDaSJEmSJGkABvVH+jnAm5O8uaoqyd5VdRVwGzDe0IbrgQVJfr+q/gs4DLhgHdUrSZI0q677wa3sc/TJgy5DkuasZX6T0wZnbV1CMpnj6N3zYmW7keZxbf55wK59N/G8X7u04zXAaUlW0RtJ8Yl1WLMkSZIkSRqQ9G5Noemav93Otcthxw66DEmSZmwtfXI12SWfajyXkKQ14wiM9dqY5xODGoEhSZIkSZI0ZQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJc1qS1Wu5/0VJHrE2tyFJkiZngCFJkjSOJBsBiwADDEmSBswAQ5IkrRfSc3ySq5OsSnJIm799kguTLG/LntrmfzzJcJJrkhzb189NSY5J8i3gVcAQcEpbf9O27IrW15Ikaeudn+QDSS5P8p2R7UiSpNlhgCFJktYXLwUWAnsBzwCOT7I98GrgnKoaWba8tX9HVQ0BewJPS7JnX1+/qqr9q+qzwDBwaFUtrKo7gROq6olVtTuwKfD8vvXmVdW+wF8C7x5dYJIjW2gyfM8dt83WfkuStEEwwJAkSeuL/YHPV9W9VfVj4ALgicAVwGuSLAb2qKqR5OAVSa4ErgJ2A3bt6+vUCbZzUJLLkqwCnt7WHXFGe1wGLBi9YlUtqaqhqhqat9kW095BSZI2ZAYYkiRpfZGxZlbVhcABwM3AZ5IcnmRn4K3AH1bVnsBXgE36Vrt9zA0kmwAfAw6uqj2AT45a7672eC8wbw32RZIkjWKAIUmS1hcXAock2SjJNvRCi8uT7AT8pKo+Cfwb8ATgt+mFFL9Isi3wnAn6vQ0YGS4xElbckmRz4OC1sB+SJGkMfjIgSZLWF2cC+wErgALeVlX/k+QI4OgkdwOrgcOr6ntJrgKuAW4ELp6g36XAJ5Lc2fr/JLAKuIne5SmSJGkdMMCQJElzWlVt3h4LOLr99C8/CThpjPUWjdPfglHPvwR8qW/WO9vP6PUO7Ju+hTHugSFJkmbOS0gkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1nt9CIkmSNACP32Frho8/fNBlSJI0ZzgCQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfO8iackSdIAXPeDW9nn6JMHXYYkSTO2bB3fjNoRGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSVovJFmY5LnrYDsHJvnyJG0WJTlhbdciSdKGxABDkiStLxYCsxJgJNloNvqRJEmzxwBDkiR1RpIFSa5P8qkkVyc5Jckzklyc5LtJ9k0yP8mJSa5IclWSFyV5KPB3wCFJlic5JMnDkpyVZGWSS5Ps2baxOMlnknyz9fn6Nv/AJOcl+RywKskmST6dZFXbzkFj1DvmNiRJ0uybN+gCJEmSRvl94OXAkcAVwKuB/YEXAn8LXAt8s6pem2Qr4HLgG8AxwFBVvQkgyUeBq6rqxUmeDpxMb5QGwJ7Ak4D5wFVJvtLm7wvsXlXfS/JXAFW1R5JdgK8leeyoWo+dYBu/IcmRbb946BZbz+DQSJK04TLAkCRJXfO9qloFkOQa4NyqqiSrgAXADsALk7y1td8E2HGMfvYHXgZQVd9MsnWSLduyf6+qO4E7k5xHL7j4X+Dyqvpe3/ofbetfn+S/gdEBxkTb+A1VtQRYAjB/u51rSkdDkiQBBhiSJKl77uqbvq/v+X30zl3uBV5WVTf0r5TkD0b1kzH6rlGPo+ffPsn6o020DUmSNIsMMGbo8TtszfDxhw+6DEmSNkTnAG9O8uY2MmPvqroKuA3Yoq/dhcChwHFJDgRuqapfJgF4UZL30buE5EDg7fzm6IqR9b/ZLh3ZEbgB2G+K25AkSbPIm3hKkqS55jhgY2Blkqvbc4DzgF1HbuIJLAaGkqwE3g8c0dfH5cBXgEuB46rqh2Ns52PARu3SlVOBRVV116g2E21DkiTNolQ5ynEmhoaGanh4eNBlSJLUNZ0fepBkMbC6qj40yDrmb7dz7XLYsYMsQZKkNbJs7V2VMOb5hCMwJEmSJElS53kPDEmStEGpqsWDrkGSJE2fIzAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmd5z0wJEmSBuDxO2zN8Nq7e7skSesdR2BIkiRJkqTOS1UNuoY5KcltwA2DrmMD93DglkEXsQHz+A+er8FgefzHdktVPXvQRcwFnkvMCn8P14zHb815DNeMx2/Nra/HcMzzCS8hmbkbqmpo0EVsyJIM+xoMjsd/8HwNBsvjr1ngucQa8vdwzXj81pzHcM14/NbchnYMvYREkiRJkiR1ngGGJEmSJEnqPAOMmVsy6ALkazBgHv/B8zUYLI+/1pTvoTXnMVwzHr815zFcMx6/NbdBHUNv4ilJkiRJkjrPERiSJEmSJKnzDDAkSZIkSVLnGWDMQJJnJ7khyX8lefug61nfJTkxyU+SXN0372FJvp7ku+3xdwZZ4/ouyaOSnJfkuiTXJPmLNt/XYR1IskmSy5OsaMf/2Dbf478OJdkoyVVJvtyee/w1Y55LrJkkNyVZlWR5kuFB1zMXeD615sY5houT3Nzei8uTPHeQNXaZ55NrZoLjt0G9Bw0wpinJRsC/AM8BdgVelWTXwVa13lsKPHvUvLcD51bVY4Bz23OtPfcAf1VVjweeBPx5e9/7OqwbdwFPr6q9gIXAs5M8CY//uvYXwHV9zz3+mhHPJWbNQVW1sKqGBl3IHLEUz6fW1FJ+8xgC/GN7Ly6sqv9cxzXNJZ5Prpnxjh9sQO9BA4zp2xf4r6q6sap+DXwBeNGAa1qvVdWFwM9GzX4RcFKbPgl48bqsaUNTVT+qqivb9G30/oh7JL4O60T1rG5PN24/hcd/nUmyA/A84FN9sz3+minPJbTOeT615sY5hpoizyfXzATHb4NigDF9jwS+3/f8B2yAb5wO2LaqfgS9X2bgdwdczwYjyQJgb+AyfB3WmXb5wnLgJ8DXq8rjv279E/A24L6+eR5/zZTnEmuugK8lWZbkyEEXM4f579jseFOSle0SEy9/mALPJ9fMqOMHG9B70ABj+jLGPL+LVhuEJJsDXwL+sqp+Oeh6NiRVdW9VLQR2APZNsvuAS9pgJHk+8JOqWjboWrTe8FxizT2lqp5A7zKcP09ywKAL0gbr48Cj6V3i+SPgHwZazRzg+eSaGeP4bVDvQQOM6fsB8Ki+5zsAPxxQLRuyHyfZHqA9/mTA9az3kmxM7x/LU6rqjDbb12Edq6r/Bc6ndw2ux3/deArwwiQ30Rvq//Qkn8Xjr5nzXGINVdUP2+NPgDPpXZaj6fPfsTVUVT9uHzLcB3wS34sT8nxyzYx1/Da096ABxvRdATwmyc5JHgq8Ejh7wDVtiM4GjmjTRwD/PsBa1ntJAvwbcF1Vfbhvka/DOpBkmyRbtelNgWcA1+PxXyeq6m+qaoeqWkDv3/xvVtUf4/HXzHkusQaSzE+yxcg08Czg6onX0jj8d2wNjfzh3bwE34vj8nxyzYx3/Da092CqHLE4Xe2raf4J2Ag4sareM9iK1m9JPg8cCDwc+DHwbuAs4IvAjsD/B7y8qryp0lqSZH/gImAVD9wD4G/pXXfn67CWJdmT3k2tNqIXPH+xqv4uydZ4/NepJAcCb62q53v8tSY8l5i5JL9Hb9QFwDzgcx6/yXk+tebGOYYH0hu6X8BNwJ+O3M9BD+b55JqZ4Pi9ig3oPWiAIUmSJEmSOs9LSCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDc06SrZMsbz//k+TmvucPnWGfByZ58mzXuja3k+SmJBeNmrc8ydXT7GdpkoPb9KeS7DqDWhYmeW7f8xcmeft0+xmn73tH9ivJaUk2m8a6j0hy+jS3d36SoTHmH5jky6PmTevY9befpN3IPq9IcuXIeybJgum+vhNsY7z93DjJ+5N8tx3zy5M8py1bPc1tLE7y1hnU9qD309reniRJkuaGeYMuQJquqroVWAi9P1iA1VX1oZHlSeZV1T3T7PZAYDVwyexUuc62s0WSR1XV95M8fk07q6rXzXDVhcAQ8J+tn7OBs9e0nubOqloIkOQU4A3Ahydbqb0PfghMGhjMhjU4dmPp3+c/At4HPG0W+5/IccD2wO5VdVeSbdfhtkcspO/91G+Gv9+SJElaDzgCQ+uF9sn2h5OcB3wgyaOTfDXJsiQXJdmltXtBksuSXJXkG0m2TbKA3h/Fb2mfej+19ffxJOcluTHJ05KcmOS6JEv7tvusJN9un5KflmTzNv+mJMe2+auS7DLWdmZh178IHNKmXwV8vq+2jZIcn+SKJCuT/GmbnyQnJLk2yVeA3+1b5/5P5ZM8u9W/Ism5bd6+SS5px++SJI9Lb9TL3wGHtP06JMmiJCe0dXZKcm6r4dwkO/a9Zh9p/dw4lZEJwEXA7yeZ316PK1otL2p9Lmqvw38AX+sfsZBkkySfbq/HVUkOavM3TfKFVt+pwKYzeSFGHbs/SfKdNu+TI8eiOWCa+/zbwM/H2N54+zPj/UxvdMvrgTdX1V0AVfXjqvpiX5v3tPfEpemFG+O+xqP6Hu938uXpjfRYkeTCcd5Pi5MsSfI14OSpbE+SJEnrH0dgaH3yWOAZVXVv+4P7DVX13SR/AHwMeDrwLeBJVVVJXge8rar+Kskn6BvJkeRPgN9p67wQ+A/gKcDrgCuSLAR+ALyzbfP2JH8N/B96f3wB3FJVT0jyRuCtVfW60dvp1/7Q/Mcx9uuOqhrvspPTgaXAh4AXAIcCh7VlfwL8oqqemOS3gIvbH4B7A48D9gC2Ba4FThxVyzbAJ4EDqup7SR7WFl3f5t2T5BnAe6vqZUmOAYaq6k1t/UV93Z0AnFxVJyV5LfAR4MVt2fbA/sAu9EZsjHu5R5J5wHOArwLvAL5ZVa9NshVweZJvtKb7AXtW1c/SC41G/DlAVe3R/nj+WpLHAn9G7xjvmWRP4MrxagCemmR53/MdgdGXlTwCeBfwBOA24JvAir4mU9nnTdt2Nmntnz5Gm/H2Z0328/eB/6+qfjnO/s8HLq2qdyT5IL2w4++Z+DUesYSxfyePAf6oqm5OslVV/XqM99NiYB9g/6q6swVUk21PkiRJ6xkDDK1PTmvhxebAk4HTkows+632uANwapLtgYcC35ugv/9oQccq4MdVtQogyTXAgtbXrvSCAVp/3+5b/4z2uAx46WTFV9V5tEtjpuFnwM+TvBK4Drijb9mzgD37PuXfEngMcADw+aq6F/hhkm+O0e+TgAur6nuttp/19XFSkscABWw8hRr344H9/wzwwb5lZ1XVfcC1I5/mj2HTvtDgIuDf6F2C88I8cL+DTeiFCQBf76u33/7AR9v+XJ/kv+mFXgfQ+wOYqlqZZOUE+3JRVT1/5En6RuP02Re4YKSGJKe17YyYyj73X0KyH71RB7tPcX9mYz/H82seCGyWAc9s0xO9xkzyO3kxsDTJF3ngd2YsZ1fVnVPZniRJktZPBhhan9zeHh8C/O/IH4CjfBT4cFWdneRAYPEE/d3VHu/rmx55Pg+4l94fy6+aZP17mcLv2gxHYACcCvwLsGh0l/QuBThn1HaeSy98mLCccdocB5xXVS9poxvOn6SfsfT3239cM7phc+fo1zK9v4JfVlU3jJr/BzzwPhhtvP5H1zTS10uAd7en07m/xUTbgant8/2q6ttJHg5sM8XtTGs/R/kvYMckW1TVbWMsv7uqRvqY6H09ejvj/k5W1Rva6/Y8YHkb3TSW8V7XsbYnSZKk9ZD3wNB6pw1//16Sl8P993zYqy3eEri5TR/Rt9ptwBbT3NSlwFOS/H7bzmZtqP5Ext1OVZ1XVQvH+JnsW0vOpPcJ9Dmj5p8D/FmSjVt9j00yH7gQeGV698jYHjhojD6/DTwtyc5t3ZFLSPqP36Kp7Be90RKvbNOH0ruMZ02dA7y5BRkk2XsK61zYtk97nXYEbhg1f3dgT4CqOrPvNRieRm2X0zt2v9Mue3nZNNb9De0ykI2AW2ewP5PuZ7+quoPeCJePtHtRkGT7JH88SZkTvsYT/U4meXRVXVZVxwC3AI9i8t/HtfGekiRJUscZYGh9dSjwJ0lWANcAL2rzF9Mbxn4RvT+WRvwH8JJM4+aaVfVTen/Ef74Nx7+U3n0NJjLt7Uyhjtuq6gNV9etRiz5F7/4WV6Z3I8t/pfeJ+ZnAd4FVwMeBC8bo86fAkcAZ7Rie2hZ9EHhfkovp/VE94jxg17Zfhzy4N44CXtOO0WHAX8x8b+93HL3LV1a2fTtuCut8DNioXRJ0KrCo3ajy48Dmrb630QsgZqyqbgbeC1wGfIPea/CLaXazaTuWy1utR7RLfvqNtz9rup/vBH5K7xKXq4Gz2vOJTOU1Hu938vj0bjh6Nb2QZQUTv5+muj1JkiStZ/LAaGBJ0mxIsnlVrW4jMM4ETqyqMwddlyRJkjSXOQJDkmbf4jZ64mp6N4o9a6DVSJIkSesBR2BIkiRJkqTOcwSGJEmSJEnqPAMMSZIkSZLUefMGXcBc9exnP7u++tWvDroMSZK6JoMuQJIkrZ8cgTFDt9xyy+SNJEmSJEnSrDDAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdV6qatA1zEnzt9u5djns2Gmvt+z4w9dCNZIkdYY38ZQkSWuFIzAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUufNWoCR5FNJdm3Tfztb/U5hu4uSPGK67frrlSRJkiRJ3TZrAUZVva6qrm1P11mAASwCJg0wRrcbVa8kSZIkSeqwaQcYSRYkuT7JSUlWJjk9yWZJzk8ylOT9wKZJlic5pa1zVpJlSa5JcmSb92dJPtjX76IkH52g/UZJlia5OsmqJG9JcjAwBJzStrdpkmOSXNHaLUnPWO3OTzLU+l6d5D1JViS5NMm2a3hcJUmSJEnSLJrpCIzHAUuqak/gl8AbRxZU1duBO6tqYVUd2ma/tqr2oRciHJVka+B04KV9fR4CnDpB+4XAI6tq96raA/h0VZ0ODAOHtu3dCZxQVU+sqt2BTYHnj9Ou33zg0qraC7gQeP1YO53kyCTDSYbvueO2aR80SZIkSZI0MzMNML5fVRe36c8C+0/S/qgkK4BLgUcBj6mqnwI3JnlSCygeB1w8XnvgRuD3knw0ybPpBSdjOSjJZUlWAU8HdpvC/vwa+HKbXgYsGKtRVS2pqqGqGpq32RZT6FaSJEmSJM2GeTNcryZ5fr8kBwLPAParqjuSnA9s0hafCrwCuB44s6pqvPZV9fMkewF/BPx5W++1o7a1CfAxYKiqvp9kcd+2JnJ3VY3sw73M/LhIkiRJkqS1YKYjMHZMsl+bfhXwrVHL706ycZveEvh5CyN2AZ7U1+4M4MWtj1Mnap/k4cBDqupLwLuAJ7T2twEjwyFGwopbkmwOHNy3rf52kiRJkiRpDplpgHEdcESSlcDDgI+PWr4EWNlu4vlVYF5rexy9y0IAqKqfA9cCO1XV5W32eO0fCZyfZDmwFPibNn8p8Ik2/y7gk8Aq4Czgir6a7m+XZNMZ7rckSZIkSRqAPHDlxBRXSBYAX243ydxgzd9u59rlsGOnvd6y4w9fC9VIktQZGXQBkiRp/TTTERiSJEmSJEnrzLRvVllVNwEb9OgLSZIkSZK0bjkCQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnpaoGXcOcNDQ0VMPDw4MuQ5KkrsmgC5AkSesnR2BIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzktVDbqGOWn+djvXLocdO+gy5rxlxx8+6BIkSbMrgy5AkiStnxyBIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6b60GGElWr83++7bztzNpl+SStVORJEmSJEmaTevLCIwpBRij21XVk9dCLZIkSZIkaZatkwAjPccnuTrJqiSHtPmnJnluX7ulSV6WZEGSi5Jc2X6e3JZvn+TCJMtbX09N8n5g0zbvlNburCTLklyT5Mg2b6x2q9vjgUnOT3J6kuuTnJIk6+LYSJIkSZKkyc1bR9t5KbAQ2At4OHBFkguBLwCHAP+Z5KHAHwJ/BgR4ZlX9KsljgM8DQ8CrgXOq6j1JNgI2q6qLkrypqhb2be+1VfWzJJu2bX2pqt4+Rrt+ewO7AT8ELgaeAnyrv0ELQ44EeOgWW6/ZEZEkSZIkSVO2ri4h2R/4fFXdW1U/Bi4Angj8X+DpSX4LeA5wYVXdCWwMfDLJKuA0YNfWzxXAa5IsBvaoqtvG2d5RSVYAlwKPAh4zhRovr6ofVNV9wHJgwegGVbWkqoaqamjeZltMZb8lSZIkSdIsWFcBxpiXY1TVr4DzgT+iNxLjC23RW4Af0xuxMQQ8tLW/EDgAuBn4TJLDf2NDyYHAM4D9qmov4CpgkynUeFff9L2su9EpkiRJkiRpEusqwLgQOCTJRkm2oRdCXN6WfQF4DfBU4Jw2b0vgR200xGHARgBJdgJ+UlWfBP4NeEJrf3eSjfvW/XlV3ZFkF+BJfXX0t5MkSZIkSXPEugowzgRWAiuAbwJvq6r/acu+Ri/Q+EZV/brN+xhwRJJLgccCt7f5BwLLk1wFvAz45zZ/CbCy3Zzzq8C8JCuB4+hdRsIY7SRJkiRJ0hyRqhp0DXPS/O12rl0OO3bQZcx5y47/jauAJElzm9/iJUmS1op1NQJDkiRJkiRpxgwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeq8VNWga5iThoaGanh4eNBlSJLUNRl0AZIkaf3kCAxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeamqQdcwJ83fbufa5bBjB12GpmHZ8YcPugRJ2hBk0AVIkqT1kyMwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnzfkAI8nCJM+dbrskL0zy9rVbnSRJkiRJmg0zCjDS05XwYyEwaYAxul1VnV1V719LNUmSJEmSpFk05RAiyYIk1yX5GHAl8G9Jrk6yKskhrc32SS5Msrwte2qbv7qvn4OTLE2yZZKbRoKQJJsl+X6SjZO8PskVSVYk+VKSzVqbl7d+V7TtPBT4O+CQts1Dkuyb5JIkV7XHx43TblGSE1q/S5N8pLW/McnBs3N4JUmSJEnSbJjuKIrHAScDfw/sAOwFPAM4Psn2wKuBc6pqYVu2fLyOquoXwArgaW3WC9q6dwNnVNUTq2ov4DrgT1qbY4A/avNfWFW/bvNOraqFVXUqcD1wQFXt3Za9d5x2o20P7A88HxhzZEaSI5MMJxm+547bJjtWkiRJkiRplkw3wPjvqrqU3h/6n6+qe6vqx8AFwBOBK4DXJFkM7FFVk/2VfypwSJt+ZXsOsHuSi5KsAg4FdmvzLwaWJnk9sNE4fW4JnJbkauAf+9adzFlVdV9VXQtsO1aDqlpSVUNVNTRvsy2m2K0kSZIkSVpT0w0wbm+PGWthVV0IHADcDHwmyeEji/qabdI3fTbwnCQPA/YBvtnmLwXeVFV7AMeOrFNVbwDeCTwKWJ5k6zHKOA44r6p2pzeqY5Mx2ozlrr7pMfdPkiRJkiQNxkxvxHkhvftJbJRkG3qhxeVJdgJ+UlWfBP4NeEJr/+Mkj2/3u3jJSCdVtRq4HPhn4MtVdW9btAXwoyQb0xuBAUCSR1fVZVV1DHALvSDjttZ+xJb0AhSARX3zR7eTJEmSJElzxEwDjDOBlfTuYfFN4G1V9T/AgfRGRlwFvIxeMAHwduDLre2PRvV1KvDHPHD5CMC7gMuAr9O7p8WI49tNQ6+mF6KsAM4Ddh25OSfwQeB9SS7mwZeZjG4nSZIkSZLmiFTV5K30G+Zvt3Ptctixgy5D07Ds+MMnbyRJWlNehilJktaKmY7AkCRJkiRJWmcMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqvFTVoGuYk4aGhmp4eHjQZUiS1DUZdAGSJGn95AgMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfNSVYOuYU6av93Otcthxw66DGlOWHb84YMuQdK6k0EXIEmS1k+OwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzpvzAUaShUmeO912SV6Y5O1rtzpJkiRJkjQb1nqAkWSjtbyJhcCkAcbodlV1dlW9fy3VJEmSJEmSZtEaBRhJFiS5PslJSVYmOT3JZkluSnJMkm8BL0/y+iRXJFmR5EutzZat3UNaX5sl+X6Sjcdq39q8PMnVbf6FSR4K/B1wSJLlSQ5Jsm+SS5Jc1R4fN067RUlOaP0uTfKR1v7GJAev0VGVJEmSJEmzajZGYDwOWFJVewK/BN7Y5v+qqvavqi8AZ1TVE6tqL+A64E+q6hfACuBprf0LgHOq6u6x2rc2xwB/1Oa/sKp+3eadWlULq+pU4HrggKrauy177zjtRtse2B94PjDmyIwkRyYZTjJ8zx23zfBwSZIkSZKk6ZqNAOP7VXVxm/4svRAAoD8k2D3JRUlWAYcCu/W1OaRNv7JvnfHaXwwsTfJ6YLxLU7YETktyNfCPfetO5qyquq+qrgW2HatBVS2pqqGqGpq32RZT7FaSJEmSJK2p2Qgwapznt/fNWwq8qar2AI4FNmnzzwaek+RhwD7ANydqX1VvAN4JPApYnmTrMeo5DjivqnanN6pjkzHajOWuvulMcR1JkiRJkrQOzEaAsWOS/dr0q4BvjdFmC+BHSTamN6ICgKpaDVwO/DPw5aq6d6L2SR5dVZdV1THALfSCjNta+xFbAje36UV980e3kyRJkiRJc8RsBBjXAUckWQk8DPj4GG3eBVwGfJ3ePSr6nQr8MQ++5GS89scnWdUuD7mQ3j00zgN2Hbk5J/BB4H1JLubBl5mMbidJkiRJkuaIVI2+AmQaKycL6I2c2H3WKpoj5m+3c+1y2LGDLkOaE5Ydf/igS5C07ngZpiRJWitmYwSGJEmSJEnSWjVvTVauqpuADW70hSRJkiRJWrccgSFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp81JVg65hThoaGqrh4eFBlyFJUtdk0AVIkqT1kyMwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM5LVQ26hjlp/nY71y6HHTvoMiRpQsuOP3zQJWjDk0EXIEmS1k+OwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzpvVACPJp5LsOpt9TmGbWyV543TbJXlEktPXbnWSJEmSJGk2zGqAUVWvq6prZ7PPKdgKmDTAGN2uqn5YVQevpZokSZIkSdIsmnGAkWR+kq8kWZHk6iSHJDk/yVBbvjrJB5IsS/KNJPu25TcmeWFrc1mS3fr6PD/JPq3tJUmuao+Pa8t3S3J5kuVJViZ5DPB+4NFt3vFJNk9ybpIrk6xK8qLW/eh2C5Jc3fpdlOSMJF9N8t0kH5zpcZEkSZIkSbNv3hqs+2zgh1X1PIAkWwJ/1rd8PnB+Vf11kjOBvweeCewKnAScDXwBeAXw7iTbA4+oqmVJfhs4oKruSfIM4L3Ay4A3AP9cVackeSiwEfB2YPeqWtjqmAe8pKp+meThwKVJzh6j3YJR+7MQ2Bu4C7ghyUer6vtrcHwkSZIkSdIsWZMAYxXwoSQfAL5cVRcl6V/+a+CrfW3vqqq7k6wCFrT5XwS+DrybXpBxWpu/JXBSG2FRwMZt/reBdyTZATijqr47apsAAd6b5ADgPuCRwLZT2J9zq+oXAEmuBXYCHhRgJDkSOBLgoVtsPYUuJUmSJEnSbJjxJSRV9R1gH3rhxPuSHDOqyd1VVW36PnojG6iq+2jBSVXdDNyaZE/gEHojMgCOA86rqt2BFwCbtPafA14I3Amck+TpY5R2KLANsE8bbfHjkfUncVff9L2MEe5U1ZKqGqqqoXmbbTGFLiVJkiRJ0myY8QiMJI8AflZVn02yGlg0w66+ALwN2LKqVrV5WwI3t+n7+03ye8CNVfWRNr0nsALoTxO2BH7SRnscRG8kBcBto9pJkiRJkqQ5Yk2+hWQP4PIky4F30LvHxUycDryS3uUkIz5Ib1THxfTuczHiEODqts1dgJOr6lbg4nYj0eOBU4ChJMP0RmNcDzBGO0mSJEmSNEfkgas8NB3zt9u5djns2EGXIUkTWnb84YMuQRue37g5lSRJ0mxYkxEYkiRJkiRJ64QBhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfMMMCRJkiRJUucZYEiSJEmSpM4zwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdl6oadA1z0tDQUA0PDw+6DEmSuiaDLkCSJK2fHIEhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOs8AQ5IkSZIkdV6qatA1zEnzt9u5djns2EGXIUnSjC07/vC10W3WRqeSJEmOwJAkSZIkSZ1ngCFJkiRJkjrPAEOSJEmSJHWeAYYkSZIkSeo8AwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnTcrAUaSv52Nfma47UVJHjHddkk+lWTXtVudJEmSJEmaDbM1AmNaAUZ6Zmvbi4BJA4zR7arqdVV17SzVIEmSJEmS1qJphwhJzkqyLMk1SY5M8n5g0yTLk5zS2vyfJFe3n79s8xYkuS7Jx4ArgXcl+WBfv4uSfHSsbbR5GyVZ2vpcleQtSQ4GhoBT2vY3TXJMkitauyUtLBmr3flJhlrfq5O8J8mKJJcm2XZNDqokSZIkSZpdMxkF8dqq2odeIHAUcDxwZ1UtrKpDk+wDvAb4A+BJwOuT7N3WfRxwclXtDXwMeGlfv4cAp461jSRbAwuBR1bV7lW1B/DpqjodGAYObdu/Ezihqp5YVbsDmwLPH6ddv/nApVW1F3Ah8PqxdrwFNsNJhu+547YZHDpJkiRJkjQTMwkwjkqyArgUeBTwmFHL9wfOrKrbq2o1cAbw1Lbsv6vqUoCq+ilwY5IntYDiccDFE2zjRuD3knw0ybOBX45T30FJLkuyCng6sNsU9unXwJfb9DJgwViNqmpJVQ1V1dC8zbaYQreSJEmSJGk2zJtO4yQHAs8A9quqO5KcD2wyutkEXdw+6vmpwCuA6+mFHjXeNqrq50n2Av4I+PO23mtH1bcJvZEdQ1X1/SSLx6hvLHdXVbXpe5nmcZEkSZIkSWvXdEdgbAn8vAULu9C7RATg7iQbt+kLgRcn2SzJfOAlwEXj9HcG8GLgVTxw+ciY20jycOAhVfUl4F3AE1r724CR4RAjYcUtSTYHDu7bVn87SZIkSZI0h0x3pMFXgTckWQncQO8SD4AlwMokV7b7YCwFLm/LPlVVVyVZMLqzNqriWmDXqhppP942Hgl8uu/bS/6mPS4FPpHkTmA/4JPAKuAm4Iq+zY1uJ0mSJEmS5og8cOWEpmP+djvXLocdO+gyJEmasWXHH742up3oUlJJkqQZm8lNPCVJkiRJktYpAwxJkiRJktR5BhiSJEmSJKnzDDAkSZIkSVLnGWBIkiRJkqTOM8CQJEmSJEmdZ4AhSZIkSZI6zwBDkiRJkiR1ngGGJEmSJEnqPAMMSZIkSZLUeQYYkiRJkiSp8wwwJEmSJElS5xlgSJIkSZKkzjPAkCRJkiRJnWeAIUmSJEmSOi9VNega5qShoaEaHh4edBmSJHVNBl2AJElaPzkCQ5IkSZIkdZ4BhiRJkiRJ6jwDDEmSJEmS1HkGGJIkSZIkqfO8iecMJbkNuGHQdXTIw4FbBl1Eh3g8Hszj8WAejwfzeDxgfTgWt1TVswddhCRJWv/MG3QBc9gNVTU06CK6Ismwx+MBHo8H83g8mMfjwTweD/BYSJIkjc9LSCRJkiRJUucZYEiSJEmSpM4zwJi5JYMuoGM8Hg/m8Xgwj8eDeTwezOPxAI+FJEnSOLyJpyRJkiRJ6jxHYEiSJEmSpM4zwJAkSZIkSZ1ngDEDSZ6d5IYk/5Xk7YOuZ9CS3JRkVZLlSYYHXc+6luTEJD9JcnXfvIcl+XqS77bH3xlkjevSOMdjcZKb23tkeZLnDrLGdSXJo5Kcl+S6JNck+Ys2f4N8f0xwPDbU98cmSS5PsqIdj2Pb/A3y/SFJkjQZ74ExTUk2Ar4DPBP4AXAF8KqqunaghQ1QkpuAoaq6ZdC1DEKSA4DVwMlVtXub90HgZ1X1/hZy/U5V/fUg61xXxjkei4HVVfWhQda2riXZHti+qq5MsgWwDHgxsIgN8P0xwfF4BRvm+yPA/KpanWRj4FvAXwAvZQN8f0iSJE3GERjTty/wX1V1Y1X9GvgC8KIB16QBqqoLgZ+Nmv0i4KQ2fRK9P9I2COMcjw1SVf2oqq5s07cB1wGPZAN9f0xwPDZI1bO6Pd24/RQb6PtDkiRpMgYY0/dI4Pt9z3/ABnwC3hTwtSTLkhw56GI6Ytuq+hH0/mgDfnfA9XTBm5KsbJeYbHBD4pMsAPYGLuP/b+9+QaSKojiOfw8jgqxBUNsqqNhEFoNFwwYRjQZB00aDxWwRBKNiM4g2FRbWP1stgtEiKGgUkZWZJHb3GN4dGGRmljF4H97vp8ybOwwcLr/yzsy5z3z8uR/QaD4iYhAR74ER8DozzYckSdIMNjAWF1PWWp/DOZOZp4CLwPUyQiBNegAcA1aA78DdqtX8YxGxF9gAbmTmz9r11DZlP5rNR2b+yswVYBk4HREnKpckSZLUWzYwFvcNODTxfhnYqlRLL2TmVnkdAS/oxmxaNyzz/uO5/1HleqrKzGG5UdsGHtJQRsrZBhvAk8x8Xpabzce0/Wg5H2OZ+QN4A1yg4XxIkiTNYwNjce+A4xFxJCJ2A1eAzco1VRMRS+UwPiJiCTgPfJz/rSZsAmvleg14VbGW6sY3Y8UlGslIOaTxEfApM+9NfNRkPmbtR8P5OBgR+8r1HuAc8JlG8yFJkrQTn0LyF8oj/u4DA+BxZt6pW1E9EXGU7l8XALuAp63tR0Q8A1aBA8AQuAW8BNaBw8BX4HJmNnGw5Yz9WKUbD0jgC3BtPOP/P4uIs8Bb4AOwXZZv0p370Fw+5uzHVdrMx0m6QzoHdD8orGfm7YjYT4P5kCRJ2okNDEmSJEmS1HuOkEiSJEmSpN6zgSFJkiRJknrPBoYkSZIkSeo9GxiSJEmSJKn3bGBIkiRJkqTes4EhSZIkSZJ6zwaGJEmSJEnqvd/cj4Kc2LjBhAAAAABJRU5ErkJggg==\n",
      "text/plain": [
       "<Figure size 1080x720 with 3 Axes>"
      ]
     },
     "metadata": {
      "needs_background": "light"
     },
     "output_type": "display_data"
    }
   ],
   "source": [
    "dframe_progression_rate=pd.DataFrame(dframe_demo_with_medication.groupby(['id','Treatment','drug'],as_index=False).agg({'target':max}))\n",
    "dframe_progression_rate=pd.DataFrame(dframe_progression_rate.groupby(['Treatment','drug'],as_index=False).agg({'id':pd.Series.nunique,'target': sum}))\n",
    "dframe_progression_rate.columns=['Treatment','drug','Patient Count','Progress Count']\n",
    "dframe_progression_rate['Progression Rate']=100*dframe_progression_rate['Progress Count']/sum(dframe_progression_rate['Progress Count'])\n",
    "dframe_progression_rate=dframe_progression_rate.sort_values('Progress Count',ascending=True)\n",
    "sns.color_palette(\"flare\", as_cmap=True)\n",
    "\n",
    "plt.figure(figsize=(20,20))\n",
    "g = sns.FacetGrid(dframe_progression_rate, col='Treatment', sharex=False, sharey=False, col_wrap=2, height=5, aspect=1.5)\n",
    "g.map_dataframe(sns.barplot, x='Progress Count', y='drug')\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3093b995",
   "metadata": {},
   "source": [
    "## 3. Data Engineering and Scaling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "c41b8bad",
   "metadata": {},
   "source": [
    "#### <font color='grey'>Feature Store Creation</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "f3e8474e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "300\n"
     ]
    }
   ],
   "source": [
    "# Extracting medication history\n",
    "\n",
    "dframe_medication_temp=pd.DataFrame(dframe_demo_with_medication.groupby(['id','Treatment'],as_index=False)['Medication Duration (days)'].sum())\n",
    "\n",
    "dframe_medication_temp_pivot=pd.pivot_table(dframe_medication_temp,\n",
    "                                            index='id',\n",
    "                                            columns='Treatment',\n",
    "                                            values='Medication Duration (days)',\n",
    "                                            aggfunc=max).reset_index()\n",
    "\n",
    "\n",
    "\n",
    "\n",
    "for col in ['Medication Period-Diabetes Type 2','Medication Period-High Blood Cholestrol','Medication Period-Hyper Tension']:\n",
    "    dframe_medication_temp_pivot[col]= dframe_medication_temp_pivot[col].apply(lambda x: 1 if x>0 else 0)\n",
    "\n",
    "    \n",
    "    # Extracting duration of observation period\n",
    "dframe_demo_alltestresults['t=n']=dframe_demo_alltestresults.groupby(['id'])['time'].transform(max)\n",
    "\n",
    "dframe_start_time=dframe_demo_alltestresults[dframe_demo_alltestresults['time']==0]\n",
    "\n",
    "dframe_latest_time=dframe_demo_alltestresults[dframe_demo_alltestresults['time']==dframe_demo_alltestresults['t=n']]\n",
    "\n",
    "dframe_Delta=dframe_start_time[['id','time','Creatinine','HGB','SBP','Glucose','Lipoprotein']].merge(dframe_latest_time[['id','time','Creatinine','HGB','SBP','Glucose','Lipoprotein']],how='inner',on='id')\n",
    "dframe_Delta.to_csv('output/Delta.csv')\n",
    "list_colnames=['id']\n",
    "for col in ['Creatinine','SBP','HGB','Glucose','Lipoprotein']:\n",
    "   \n",
    "    dframe_Delta[col+'_baseline']=round(dframe_Delta[col+'_x'],1)\n",
    "    dframe_Delta[col+'_lastObs']=round(dframe_Delta[col+'_y'],1)\n",
    "    list_colnames.extend([col+'_baseline',col+'_lastObs'])\n",
    "\n",
    "print(len(dframe_Delta.id.unique()))"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "e35aa993",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(300, 21)\n",
      "(300, 24)\n",
      "300\n"
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
       "      <th>race</th>\n",
       "      <th>gender</th>\n",
       "      <th>Age Group</th>\n",
       "      <th>target</th>\n",
       "      <th>time</th>\n",
       "      <th>CKD(t=0)</th>\n",
       "      <th>Dieabetes(t=0)</th>\n",
       "      <th>Cholestrol(t=0)</th>\n",
       "      <th>Hyper Tension(t=0)</th>\n",
       "      <th>...</th>\n",
       "      <th>SBP_lastObs</th>\n",
       "      <th>HGB_baseline</th>\n",
       "      <th>HGB_lastObs</th>\n",
       "      <th>Glucose_baseline</th>\n",
       "      <th>Glucose_lastObs</th>\n",
       "      <th>Lipoprotein_baseline</th>\n",
       "      <th>Lipoprotein_lastObs</th>\n",
       "      <th>Medication Period-Diabetes Type 2</th>\n",
       "      <th>Medication Period-High Blood Cholestrol</th>\n",
       "      <th>Medication Period-Hyper Tension</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>0</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>61-70</td>\n",
       "      <td>1</td>\n",
       "      <td>1196</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>147.5</td>\n",
       "      <td>13.5</td>\n",
       "      <td>13.1</td>\n",
       "      <td>6.2</td>\n",
       "      <td>5.8</td>\n",
       "      <td>161.5</td>\n",
       "      <td>157.9</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>71-80</td>\n",
       "      <td>0</td>\n",
       "      <td>1394</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>149.7</td>\n",
       "      <td>13.9</td>\n",
       "      <td>12.8</td>\n",
       "      <td>10.0</td>\n",
       "      <td>9.8</td>\n",
       "      <td>89.6</td>\n",
       "      <td>73.2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>2</td>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>51-60</td>\n",
       "      <td>1</td>\n",
       "      <td>1254</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>136.5</td>\n",
       "      <td>15.3</td>\n",
       "      <td>15.2</td>\n",
       "      <td>7.2</td>\n",
       "      <td>7.0</td>\n",
       "      <td>61.6</td>\n",
       "      <td>87.1</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>3</td>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>71-80</td>\n",
       "      <td>0</td>\n",
       "      <td>1414</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>142.9</td>\n",
       "      <td>13.7</td>\n",
       "      <td>12.7</td>\n",
       "      <td>6.4</td>\n",
       "      <td>5.8</td>\n",
       "      <td>99.7</td>\n",
       "      <td>101.4</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>4</td>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>40-50</td>\n",
       "      <td>1</td>\n",
       "      <td>1080</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>166.4</td>\n",
       "      <td>13.2</td>\n",
       "      <td>11.6</td>\n",
       "      <td>8.8</td>\n",
       "      <td>11.0</td>\n",
       "      <td>65.7</td>\n",
       "      <td>72.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
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
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>71-80</td>\n",
       "      <td>1</td>\n",
       "      <td>1157</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>134.8</td>\n",
       "      <td>13.7</td>\n",
       "      <td>14.3</td>\n",
       "      <td>6.5</td>\n",
       "      <td>5.8</td>\n",
       "      <td>111.7</td>\n",
       "      <td>123.8</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>296</th>\n",
       "      <td>296</td>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>61-70</td>\n",
       "      <td>0</td>\n",
       "      <td>1159</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>143.2</td>\n",
       "      <td>15.2</td>\n",
       "      <td>15.3</td>\n",
       "      <td>7.2</td>\n",
       "      <td>7.9</td>\n",
       "      <td>100.4</td>\n",
       "      <td>91.8</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>297</td>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>71-80</td>\n",
       "      <td>1</td>\n",
       "      <td>1008</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>136.8</td>\n",
       "      <td>14.0</td>\n",
       "      <td>14.4</td>\n",
       "      <td>8.3</td>\n",
       "      <td>9.6</td>\n",
       "      <td>135.0</td>\n",
       "      <td>152.1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>298</td>\n",
       "      <td>Asian</td>\n",
       "      <td>Female</td>\n",
       "      <td>71-80</td>\n",
       "      <td>0</td>\n",
       "      <td>877</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>...</td>\n",
       "      <td>133.4</td>\n",
       "      <td>15.3</td>\n",
       "      <td>17.2</td>\n",
       "      <td>6.4</td>\n",
       "      <td>6.0</td>\n",
       "      <td>79.1</td>\n",
       "      <td>62.1</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0.0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>299</td>\n",
       "      <td>Asian</td>\n",
       "      <td>Male</td>\n",
       "      <td>81 and Above</td>\n",
       "      <td>0</td>\n",
       "      <td>1295</td>\n",
       "      <td>2.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>...</td>\n",
       "      <td>113.4</td>\n",
       "      <td>14.8</td>\n",
       "      <td>14.5</td>\n",
       "      <td>5.6</td>\n",
       "      <td>6.2</td>\n",
       "      <td>66.4</td>\n",
       "      <td>79.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>300 rows × 24 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      id     race  gender     Age Group  target  time  CKD(t=0)  \\\n",
       "0      0  Unknown    Male         61-70       1  1196       3.0   \n",
       "1      1    White  Female         71-80       0  1394       4.0   \n",
       "2      2    White  Female         51-60       1  1254       4.0   \n",
       "3      3    White    Male         71-80       0  1414       1.0   \n",
       "4      4    White  Female         40-50       1  1080       3.0   \n",
       "..   ...      ...     ...           ...     ...   ...       ...   \n",
       "295  295    White  Female         71-80       1  1157       4.0   \n",
       "296  296    White  Female         61-70       0  1159       4.0   \n",
       "297  297  Unknown    Male         71-80       1  1008       3.0   \n",
       "298  298    Asian  Female         71-80       0   877       3.0   \n",
       "299  299    Asian    Male  81 and Above       0  1295       2.0   \n",
       "\n",
       "     Dieabetes(t=0)  Cholestrol(t=0)  Hyper Tension(t=0)  ...  SBP_lastObs  \\\n",
       "0                 0                1                   1  ...        147.5   \n",
       "1                 1                0                   0  ...        149.7   \n",
       "2                 1                0                   1  ...        136.5   \n",
       "3                 0                0                   1  ...        142.9   \n",
       "4                 1                0                   0  ...        166.4   \n",
       "..              ...              ...                 ...  ...          ...   \n",
       "295               0                1                   1  ...        134.8   \n",
       "296               1                1                   0  ...        143.2   \n",
       "297               1                1                   0  ...        136.8   \n",
       "298               0                0                   1  ...        133.4   \n",
       "299               0                0                   0  ...        113.4   \n",
       "\n",
       "     HGB_baseline  HGB_lastObs  Glucose_baseline  Glucose_lastObs  \\\n",
       "0            13.5         13.1               6.2              5.8   \n",
       "1            13.9         12.8              10.0              9.8   \n",
       "2            15.3         15.2               7.2              7.0   \n",
       "3            13.7         12.7               6.4              5.8   \n",
       "4            13.2         11.6               8.8             11.0   \n",
       "..            ...          ...               ...              ...   \n",
       "295          13.7         14.3               6.5              5.8   \n",
       "296          15.2         15.3               7.2              7.9   \n",
       "297          14.0         14.4               8.3              9.6   \n",
       "298          15.3         17.2               6.4              6.0   \n",
       "299          14.8         14.5               5.6              6.2   \n",
       "\n",
       "     Lipoprotein_baseline  Lipoprotein_lastObs  \\\n",
       "0                   161.5                157.9   \n",
       "1                    89.6                 73.2   \n",
       "2                    61.6                 87.1   \n",
       "3                    99.7                101.4   \n",
       "4                    65.7                 72.0   \n",
       "..                    ...                  ...   \n",
       "295                 111.7                123.8   \n",
       "296                 100.4                 91.8   \n",
       "297                 135.0                152.1   \n",
       "298                  79.1                 62.1   \n",
       "299                  66.4                 79.0   \n",
       "\n",
       "     Medication Period-Diabetes Type 2  \\\n",
       "0                                  1.0   \n",
       "1                                  0.0   \n",
       "2                                  0.0   \n",
       "3                                  0.0   \n",
       "4                                  1.0   \n",
       "..                                 ...   \n",
       "295                                0.0   \n",
       "296                                1.0   \n",
       "297                                1.0   \n",
       "298                                1.0   \n",
       "299                                0.0   \n",
       "\n",
       "     Medication Period-High Blood Cholestrol  Medication Period-Hyper Tension  \n",
       "0                                        1.0                              1.0  \n",
       "1                                        1.0                              0.0  \n",
       "2                                        1.0                              0.0  \n",
       "3                                        1.0                              0.0  \n",
       "4                                        1.0                              1.0  \n",
       "..                                       ...                              ...  \n",
       "295                                      0.0                              0.0  \n",
       "296                                      1.0                              0.0  \n",
       "297                                      1.0                              1.0  \n",
       "298                                      1.0                              0.0  \n",
       "299                                      1.0                              1.0  \n",
       "\n",
       "[300 rows x 24 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# The longitudinal data needs to be aggregated in a way so that the crucial information is still retained.\n",
    "\n",
    "\n",
    "\n",
    "# The remaining static information is kep as ist\n",
    "\n",
    "original_columns=['id','race','gender','Age Group',\n",
    "                  'target']\n",
    "\n",
    "        \n",
    "dict_feature_Operations= {  'time': lambda x: x.max()-x.min(),\n",
    "                             'CKD(t=0)':sum,\n",
    "                             'Dieabetes(t=0)':sum,\n",
    "                             'Cholestrol(t=0)':sum,\n",
    "                             'Hyper Tension(t=0)':sum,\n",
    "                              'Hemoglobin(t=0)':sum,\n",
    "                              'Cholestrol(t=0)':sum\n",
    "                            \n",
    "                         }\n",
    "\n",
    "\n",
    "dframe_deltaTestResults=dframe_demo_alltestresults.groupby(original_columns,as_index=False).agg(dict_feature_Operations).merge(dframe_Delta[list_colnames],on='id',how='inner')\n",
    "print(dframe_deltaTestResults.shape)\n",
    "dframe_final=dframe_deltaTestResults.merge(dframe_medication_temp_pivot,on='id',how='left')\n",
    "dframe_final.fillna(0).to_csv(r'output/Prediction_Ready.csv',index=False)\n",
    "dframe_final=dframe_final.fillna(0)\n",
    "print(dframe_final.shape)\n",
    "print(len(dframe_final.id.unique()))\n",
    "\n",
    "dframe_final\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "a281510e",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['race', 'gender', 'Age Group', 'CKD(t=0)', 'Dieabetes(t=0)', 'Cholestrol(t=0)', 'Hyper Tension(t=0)', 'Hemoglobin(t=0)', 'Glucose_baseline', 'Lipoprotein_baseline', 'SBP_baseline', 'HGB_baseline', 'Creatinine_baseline', 'Glucose_lastObs', 'Lipoprotein_lastObs', 'SBP_lastObs', 'HGB_lastObs', 'Creatinine_lastObs', 'Medication Period-Diabetes Type 2', 'Medication Period-High Blood Cholestrol', 'Medication Period-Hyper Tension']\n"
     ]
    }
   ],
   "source": [
    "feature_set1=['race','gender','Age Group']\n",
    "feature_set2= ['CKD(t=0)','Dieabetes(t=0)','Cholestrol(t=0)','Hyper Tension(t=0)','Hemoglobin(t=0)']\n",
    "feature_set3=['Glucose_baseline','Lipoprotein_baseline','SBP_baseline','HGB_baseline','Creatinine_baseline',\n",
    "             'Glucose_lastObs','Lipoprotein_lastObs','SBP_lastObs','HGB_lastObs','Creatinine_lastObs']\n",
    "feature_set4=['Medication Period-Diabetes Type 2','Medication Period-High Blood Cholestrol','Medication Period-Hyper Tension']\n",
    "\n",
    "target='target'\n",
    "\n",
    "for f in feature_set1:\n",
    "    dframe_final[f]=dframe_final[f].astype('object')\n",
    "for f in feature_set2:\n",
    "    dframe_final[f]=dframe_final[f].astype('object')\n",
    "for f in feature_set4:\n",
    "    dframe_final[f]=dframe_final[f].astype('object')\n",
    "\n",
    "\n",
    "    \n",
    "predictors=list()\n",
    "for f in [feature_set1,feature_set2,feature_set3,feature_set4]:\n",
    "    for item in f:\n",
    "        predictors.append(item)\n",
    "        \n",
    "print(predictors)\n",
    "# my_internal_func.my_plot(dframe_final[predictors])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1f80436f",
   "metadata": {},
   "source": [
    "## 4. Pre-cursory data adjustments for modelling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "5c96c3d0",
   "metadata": {},
   "source": [
    "#### <font color='green'> Outlier Treatment</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "d2f17709",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Original data:(300, 10)\n"
     ]
    },
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
      "\n",
      "Removal of outliers==== 0\n",
      "Shape of new dataframe  (300, 10)\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAcsAAAD4CAYAAACDm83wAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjMuNCwgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy8QVMy6AAAACXBIWXMAAAsTAAALEwEAmpwYAAA7/ElEQVR4nO3de3xcVbn/8c83TUtv5EIpHMBCgQPlx7WWFA8G26IIiv6QVDioVajlRQULHDgHBQ1WrCcIgjcgihWhoKgcIVGOIHAstNWI0LT0DgEK5ZTLT4ImqUBbms7z+2OvCbvTSSaXycwked6vV1/Zs/baaz17T5pn1t579pKZ4ZxzzrnOFeU7AOecc67QebJ0zjnnMvBk6ZxzzmXgydI555zLwJOlc845l0FxvgNw2bf33nvbxIkT8x2Gc84NKCtWrHjDzManW+fJchCaOHEijY2NWW0zkUjQ1tZGIpFgy5YtAJSUlFBeXk5RkZ+gcM4NfJJe6mydJ0vXLW1tbcybN2+38traWsrLy/MQkXPO5Y4PCZxzzrkMPFk655xzGfhpWNel5LXK1tbWjHUASktL/Rqmc27Q8WTputTZtcrO6vg1TOfcYORDAJdVl156KbNmzeLxxx9nwYIFvPTSSyxYsIBNmzaxYMGCLkeozjlXqDxZuqzauXMnZsaPfvQjmpqauOWWW2hqaqK2tpampibq6uryHaJzzvVYTpKlpH0l/ULSC5JWSHpcUpWkGZJ+l4sYuqu/Y5I0W9ItYflCSef2V1/ZkEgkulzf2trKhg0bditvb2/HzHjllVd2+bls2TIfXTrnBpx+v2YpScBvgDvN7DOh7CDgDKClv/svZGZ2a75jyCT5AILOVFdX96i9RCJBXV0dc+bM6UtYzjmXU7kYWX4QeCeeGMzsJTO7OV5J0jWSroi9XidpYlg+V9IaSasl/SyUHSRpcShfLOnAUH522Ha1pGWhbJikGyQtD/W/kCHmEkn1kjZIulVSUWjnR5IaJa2X9I1YrNeFumsk3RjKxku6L/S5XFJlaifxfZa0RNL1kp6U9KykD/QkdklzQ2yNzc3NGXYvf9rb22loaMh3GM451yO5SJZHASt7u7Gko4Bq4INmdhzwb2HVLcBdZnYscDdwUyifD5wW6p4Rys4H2sxsKjAVuEDSwV10ewLwH8AxwKHAzFBebWYVwLHAdEnHStoLqAKOCrH8Z6j7A+B7oc9PArd1Y3eLzewE4DLg6z2J3cwWmlmFmVWMH5/20Ya9UlJS0uX6mpoa9tlnn263V1xcTGXlbp8bnHOuoOX8Bh9JtWHUt7ybm3wQuNfM3gAws7+H8hOBX4TlnwEnheUGYJGkC4BhoexU4FxJq4AngHHAYV30+aSZvWBmO4Ffxtr+V0krgaeIPgQcCWwBtgG3SZoJvB3qngLcEvq8n2i0umeGfU3e/bICmNjL2LMq03cmy8rKenRKtaioiJkzZ2au6JxzBSQX37NcTzSyAsDM5knaG0h90nc7uybvkeGnAOtGPxbav1DS+4CPAaskTQ5tXGJmD3cz5tT+LIzmrgCmmlmLpEXASDNrl3QC8CHgU8DFRAm+CDjRzLbGG4ou4XZqe/i5k3ffm57GnnMTJkzYray4uJidO3ey//778+qrr3b8nDZtGmVlZbkP0jnn+iAXI8tHgZGSLoqVjU5TbxMwBUDSFCB5qnEx0YhuXFi3Vyj/M1FyApgF/CmsP9TMnjCz+cAbwATgYeAiScNDncMljeki5hMkHRyuVZ4T2i4B3gLaJO0LfDS0NRYoNbMHiU6fTg5tPEKUOAn1kuU91dPY82rYsGFI4qKLLmLSpElcfPHFTJo0iXnz5jFp0iQfVTrnBqR+H1mamUk6E/iepC8DzURJ58qUqvfx7unG5cCzYfv1kmqApZJ2Ep0CnQ1cCtwu6Uuhzc+Hdm6QdBjRiGwxsBpYQ3Rac2W4O7cZOLOLsB8HriO6ZrkMqDezhKSniEbKLxCd7gXYE/itpJGhz8tD+aVAraQ1RMd5GXBh5iO2m9t6GHte3XTTTR1P8DnxxBMBmD9//i4/nXNuoJFZd85wuoGkoqLCsjWfZfzZsOm+JlJbW0tpaak/G9Y5N+BJWhFu4tyNPxvWdamoqCjjs167U8c55wayIZssJR1DdBdt3HYze18+4nHOOVe4hmyyNLO1vHszjnPOOdepIZssXc+UlpZSW1tLIpHoeAReSUkJpaWleY7MOef6nydL1y3x65Ljxo3LczRDT3yC7Vz2Gf9gVEg3bvmNZC7XPFk6NwB0ZxLuocQnGXe55h/NnHPOuQx8ZOncAPMf792HkuHDMlfsoy3v7OQ7q16P+py8DyUj+r/PLuPZsZPvPPV6XmNwQ5cnS+cGmJLhwyjdI7eJq2RE7vt0rpB4snQDQvwGF7+5w7n+4f/POudHwg0IyRtc5s2bl/O7Qp0bKvz/Wec8WTrnnOu2lpYWFixYQGtra75DySlPls4557qtvr6epqYm6urqMlceRDJes5T0ppmNTSm7EHjbzO7qt8g6j2cysH+YP7KrehXAuWZ2aQ/bnwj8zsyO7nWQXbc/A7jCzD4u6QzgSDO7rj/6GkwSiUTH8lD7RAu77nNiiM4UFN/vofg7kAu7/J7F/s8ltbS0sHTpUsyMZcuWMXPmzCEzmXuvbvAxs1uzHUicpGFmtrOT1ZOBCqDLZGlmjUB25qnqJ2Z2P3B/vuMYCJJPkgHSThU2lLy5I0H5yHxHkXtv7nj3j/dQ/x3IhS1btuz2tK76+nqS0zomEgnq6uqYM2dOPsLLuV6dhpV0jaQrwvISSd+X9GdJ6ySdEMr3kvQbSWsk/UXSsbFtfybpUUnPSboglM+Q9JikXwBrJY2UdIektZKeknSypBHAAuAcSasknSNpjKTbJS0P9T4Ra+93sT5vD7G+ICnTaLNY0p0h9nsljQ7tzA/9rJO0MEzGjKRLJW0I9X8VytLGlXIcZ0u6JSwvknRTOI4vSDorVu9LoZ01kr7RyXsyV1KjpMbm5ubuvpXOOddtDQ0NtLe3A9De3k5DQ0OeI8qdbH11ZIyZvV/SNOB24GjgG8BTZnampA8Cd/HuLB/HAv8CjAGekvRAKD8BONrMXpT0HwBmdoykI4BHgMOB+UCFmV0MIOla4FEzmyOpDHhS0h/SxHgEcDKwJ9Ak6UdmtqOT/ZkEnG9mDZJuB74I3AjcYmYLQr8/Az4O/DdwFXCwmW0PMQBUdzOuuP2Ak0Ks9wP3SjoVOCwcGwH3S5pmZsviG5rZQmAhRJM/Z+hnwCkpKelYrqmpGTKnfpLik2+PHT40bzWI7/dQ/B3IhfjvWfz/XFJlZSVLliyhvb2d4uJiKisrcx1i3mQrWf4SwMyWSSoJyeEk4JOh/FFJ4yQlp6j4rZltBbZKeowoEbQCT5rZi6HOScDNYftnJL1ElCxTnQqckRzpAiOBA9PUe8DMtgPbJb0O7Au83Mn+bDaz5EemnwOXEiXLkyV9GRgN7AWsJ0qWa4C7Jf0G+E0P44r7jZklgA2S9o21cyrwVHg9lih5Lkuz/aAV/75XWVnZkH4uaFF0QmPIie/3UP8dyIV037Gsqqpi6dKlHetnzpyZ67DyJlvJMnUkY0SjoM7qpasP8FasrLt/EQR80syadil8N9kkbY8t76Trfd8tPkkjgR8SjWo3S7qGKAECfAyYBpwBfE3SUT2Iq7MYFfv5LTP7cRfbOedcvysvL2f69OksXryYadOmDanRfbbO55wDIOkkoM3M2ohGPrNC+QzgDTNL3qXxiXBNchwwA1ieps349ocTjcqagH8QnUpNehi4JHb98L1Z2J8DJZ0Ylj8N/Il3E+MbksYCZ4X+ioAJZvYY8GWgjGj0l624HgbmhD6RdICkfXrZlnPO9UlVVRWTJk0aUqNK6N7IcrSk+OnK76ap0yLpz0AJkLw16hrgDklrgLeB82L1nwQeIEqA3zSzV0NCjPshcKuktUA7MDtcE3wMuErSKuBbwDeB7wNrQmLaRHQtsS+eBs6T9GPgOeBHZva2pJ8Aa0MfyQQ/DPh5OMUs4Htm1iopK3GZ2SOS/g/weMi7bwKfBYbUE6WTk08nl51z2ded/2fl5eXMnz8/l2EVhIzJ0sy6M/q8z8y+krLd34Hd7gANnjWzuSn1lwBLYq+3AbPTxPN3YGpK8RfS1Otoz8yuSVnX6XcozWwTcGQn664Grk6z6qQ0dbd2I65FwKKwPDul3tjY8g+AH3QW81AQn3zaOdc//P9Z5/xB6s4NMFt2dPYV5Cz3887OtMv5kqv9di6dPidLM5vRw/rX9LXPbAjXSxenWfUhM/tbruNxrrvyMadjcl5L54aqITuyDAlxcr7jcM45V/iGbLJ0biCJ33iRK4lEouMxgyUlJQU1t6Hf5OVyzZOlcwNAvm68SH026GAVn/S4J9tk88OET7Zc2DxZOueGvOSkx/lUW1vrd6IWMP8Y45xzzmXgI0vnnIuZXHk2w/cYnbHeO9vfZnXDrwE4rvJsRnRjm1Q7tr/NqtCGK2yeLJ1zORe/Rlho1+qG7zGaPUaO6dE2I3qxzUBUyO9bfxs6e+qcKxjJa4Tz5s3r8Y01Ln+G8vvmydI555zLwJOlc865XmtpaWHBggVs2rSJBQsW0Nra2lHW2tqa7/CyZsAmS0nVktZLWiNplaT3SVoiqSm8flrS3Fj9TZLWSlot6RFJ/9RF22/2Y9wTJa0LyxWSbuqvvpxzrr/V19fT1NREbW0tTU1N1NXVdZTV1dXlO7ysGZA3+IS5Jj8OTAnTdu0NjAirZ5lZo6S9gI2SFpnZO2HdyWb2hqRrga8Cl+Y++neZWSPQmM8YnMuHRCLRsVwIo494DGaJzitmWbyvQjgOmcRjTCQStLS0sHTpUsyMV155BYClS5cCYGYsW7aMmTNnDopJogdksgT2I5pMejuAmb0BEOZ7TBoLvAWkm6pgGRkSpaTvACcDLcCnzKxZ0gXAXKLE/DzwuTDP5dnA10NfbWY2TdIw4Dqiya33AGrN7McpfcwArjCzj0u6hmh+z0PCz++b2U2h3mdDvCOAJ4AvmplPweAGrOSTbwCqq6vzGMnudryzjZGj9sxcMUt9JRXacchky5YtPPbYY5jZLuXt7e0dy4lEgrq6OubMmZO6+YAzUE/DPgJMkPSspB9Kmh5bd3eYcLqJaGLpdEnl40STOHdmDLDSzKYAS4kSIUCdmU01s+OIJog+P5TPB04L5WeEsvOJEudUovk3L5B0cIb9OgI4DTgB+Lqk4WHi53OASjObTJSQZ6VuKGmupEZJjc3NzRm6cc65vmtoaNglOUI0okwm0Pb2dhoaGvIRWtYNyJGlmb0p6XjgA0Sjv3skXRVWJ0/Djgf+LOkhM3sprHtM0k5gDekncU5KAPeE5Z8DyRPvR0v6T6CMaOT6cChvABZJ+q9Y3VOBYyWdFV6XAocBz3bR7wNhtLxd0uvAvsCHgOOB5WHkPArYbb4kM1sILASoqKiw1PXOFZKSkpKO5Zqamryfpmttbe0Y2Q0fMTJn/cb7KoTjkEn8OJWUlFBZWcmSJUt2SZjJM3xmRnFxMZWVlXmJNdsGZLIECCPGJcASSWuB81LWN0taCbwPSCbLk5OnbHvaXfi5CDjTzFZLmk10ihUzu1DS+4CPAaskTQYEXGJmD8cbkjSxi362x5Z3Er0/Au40s6/0Im7nClL8y+xlZWUF9UxUKXcn3OJ9FdpxyKSoqIiqqqqOa5RJxcVRWtmxYwdFRUXMnDkzH+Fl3YA8DStpkqTDYkWTeTchJuuMBt4LbOxFF0VAckT4GeBPYXlP4DVJw4mdCpV0qJk9YWbzgTeACUSjzotCXSQdLqk3j/hYDJwlaZ/Qzl6SDupFO845l1Xl5eVMnz4dSRxwwAFIYvr06R1l06ZNK/jRcncN1JHlWOBmSWVAO9HNNnOBe4muWW4luqlmkZmt6EX7bwFHSVoBtBFdMwT4GtENNi8RXfNM3gVwQ0jeIkpuq4lO9U4EVio6L9EMnNnTQMxsg6SrgUcUfQzdAcwj5cOBc87lQ1VVFS+//DLnnnsud911FzNnzsTMePnllwfNqBJAqXcyuYGvoqLCGhv9GymucBXaM0ZbWlo6puia+sHzuvWc1+3b3mL5o3f2aJuu2hgIU3QV2vuWbZJWmFlFunUDdWTpnBvA8jWZdXfs2P52t+q9E6v3Tje36W1fhaKQ37f+NqSTpaQniE7Xxn3OzLr6WolzbhDrzZRZq32arUFvSCdLM3tfvmNwzjlX+IZ0snTOOYiuv9XW1vZom0Qi0fEkopKSkj5fvystLe3T9q5/ebJ0zg15vb0WN27cuH6IxhUiT5auXyQfspz6yXsw3kHnnBv8PFm6ftHW1sYll1yyW/lAuD3eOedS+Ud855xzLgNPls4551wGfhrWZUXqkz16UtevYTrnCp0nS5cVbW1tHY8Ly3QLfmpdv4bpnCt0/pHe9VlLSwvf+9738h2Gc871G0+Wrs/q6+vZuLE3M6E559zAMGCSpaQ3U17PlnRL7PVnJa2RtF7Sakm3hSm8kLREUpOkVZKeljS3J31lk6SJktaF5QpJN/VXX7nQ0tLC0qVLic9e09raSmtra9r6qesSiUQ/R+icc303KK5ZSvoIcDnwUTN7RdIw4DxgX6A1VJtlZo2S9gI2SlpkZu/kJ+KImTUCA3ourfr6elKneauuru60fuq6LVu2+FNQnHMFb8CMLDOoBq4ws1cAzGynmd1uZk1p6o4lmtx5Z1cNSvqOpJWSFksaH8oukLQ8jFzvkzQ6lJ8taV0oXxbKhkm6IdRfI+kLafqYIel3YfkaSbeHUfALki6N1fuspCfDyPjH4cNAaltzJTVKamxubu7uceuzhoYG2tvbc9afc87lw0BKlqNCslglaRWwILbuKGBlhu3vlrQGaAK+aWZdJcsxwEozmwIsBb4eyuvMbKqZHQc8DZwfyucDp4XyM0LZ+UCbmU0FpgIXSDo4Q4xHAKcBJwBflzRc0v8BzgEqzWwyUZKflbqhmS00swozqxg/fnyGbrKnsrKS4uJdT1DU1NRQU1OTtn7qupKSkn6NzznnsmEgJcutZjY5+Y8oQe1G0jEhoW6UdE5s1SwzOxY4ELhC0kFd9JUA7gnLPwdOCstHS/qjpLVECeuoUN4ALJJ0AZAc9Z0KnBsS+xPAOOCwDPv4gJltN7M3gNeJTiN/CDgeWB7a+hBwSIZ2cqaqqgpJu5SVlZVRVlaWtn7qOv+OpXNuIBgsf6nWA1MAzGxtSKa/B0alVjSzZqJRaE/mskxelFsEXGxmxwDfAEaGNi8ErgYmAKskjQMEXBJL8Aeb2SMZ+tkeW95JdE1ZwJ2xdiaZ2TU9iL1flZeXM3369N0SpnPODSaDJVl+C7hR0ntiZbslSoBwnfG9QFffdSgCzgrLnwH+FJb3BF6TNJzYqVBJh5rZE2Y2H3iDKGk+DFwU6iLpcEljerxnsBg4S9I+oZ29MoyKc66qqopDDz0032E451y/GRR3w5rZg+EmnN+Hm19agXVECSvpbklbgT2ARWa2oosm3wKOkrQCaCO6ZgjwNaJTqi8Ba4mSJ8ANkg4jGgUuBlYDa4CJwEpFw65m4Mxe7NsGSVcDj0gqAnYA80IMBaG8vJzLL7+846k8zjk32Cj1tn838FVUVFhjY26/kZL6vNf4I+3iamtrO9Yn6/p1S+dcIZC0wswq0q0bFCNLl389mWm+t7PSO+dcvgzpZCnpCaLTsnGfM7O1+YjHOedcYRrSydLMenJHrHPOuSFqSCdL139KS0u5+eab2bJlCxA9fKCoqCjjXJfOOVeIPFm6flFUVMS4ceP8ua/OuUHBk6Xr0N7ezubNm4FoJFheXu53qjrnHJ4sXczmzZt3mRWktrbW71p1zjkGzxN8nHPOuX7jydI555zLwJOlA6In8CTvXE1qbW0lkUjkKSLnnCscniwdAG1tbVx//fW7lFVXV3c8ls4554YyT5auS5dddhkvvVQwz2x3zrm88GTpurRjxw5uueWWfIfhnHN51atkKemfJP1K0kZJGyQ9KOnwvgQiqUzSF2Ov95d0bze2e1BSWV/6TmlvkaSzMtfsdftLJFWE5azG3hddXZt85ZVXfHTpnBvSepwsw9yM9cASMzvUzI4EvgrsG6szrBexlAEdydLMXjWzjEnLzE43s9Ze9Jd3hRR76s09qXx06ZwbynozsjwZ2GFmtyYLzGwVMEzSY5J+AayVNEzSDZKWS1oj6QsAksZKWixppaS1kj4RmrkOOFTSqrDdREnrwjazJdVJekjSc5K+nexb0iZJe4f6T0v6iaT1kh6RNCrUOTRsu0LSHyUdkWEfTwn1npX08dDGxFC2Mvx7fyjfT9KyEPc6SR8I5adKejzU/bWksamdZDN2SXMlNUpqbG5uzvgm9tQrr7yS9Tadc26g6E2yPBpY0cm6E4DqMNo8H2gzs6nAVOACSQcD24AqM5tClHi/E0arVwEbzWyymX0pTduTgXOAY4BzJE1IU+cwoNbMjgJagU+G8oXAJWZ2PHAF8MMM+zgRmA58DLhV0kjgdeDDIe5zgJtC3c8AD5vZZOA4YJWkvYGrgVNC/Ubg3zP02afYzWyhmVWYWcX48eMzdLW7kpKSLtcfcMABPW7TOecGi2w/7u5JM3sxLJ8KHBu7/ldKlBBeBq6VNA1IAAcQO4XbhcVm1gYgaQNwELA5pc6LYZQLUUKfGEZ07wd+HeVkYPc5LFP9l5klgOckvQAcAbwI3CJpMrATSF6jXQ7cLmk48BszWyVpOnAk0BD6HAE8nqHPbMXeK5meAXvxxRf3R7fOOTcg9CZZrgc6u5b4VmxZRCOih+MVJM0GxgPHm9kOSZuAkd3od3tseSfpY0+tM4po9NwaRn7dZWleXw78lWj0WEQ0QsbMloXE/zHgZ5JuAFqA/zGzT/egz2zFnnUHHHAABx10UD5DcM65vOrNadhHgT0kXZAskDSV6LRl3MPARWHEhaTDJY0hGmG+HhLlyUQjRIB/AHv2Ip4umdkW4EVJZ4c4JOm4DJudLalI0qHAIUBTiPu1MOL8HDAstHdQ2J+fAD8FpgB/ASol/XOoM7o3dwv3MvasGj58uI8qnXNDXo+TpZkZUAV8WNFXR9YD1wCvplS9DdgArAw36vyYaDR4N1AhqRGYBTwT2v0b0WnLdWF0lk2zgPMlrSYaGX8iQ/0mYCnwe+BCM9tGdK3wPEl/IToFmxxFzyC6TvkU0XXGH5hZMzAb+KWkNUTJM9NNRdmKPau+//3v+6jSOTfkKcp9bjCpqKiwxsbGHm2TSCRYu3btLo+8q6mp4aCDDvI5LZ1zQ4KkFWZWkW6d/xV0QHSDT+odsWVlZZ4onXOOITz5s6Rq4OyU4l+bWU0+4nHOOVe4hmyyDEnRE6NzzrmMhmyydLubMGECNTXR54eSkhJKS0vzHJFzzhUGT5auQ3FxMQcffHDadYlEgpaWlo5nyJaUlFBUVERpaalf13TODXqeLF23tLW1cckll+xWXltbS3l5eR4ics653PEhgXPOOZeBJ0vnnHMuA0+WzjnnXAZ+zdJ1KpFI0NbW1rHcnXp+w49zbjDyv2quU21tbcybN4958+Z13AWbqV4yaTrn3GDiydJl1eWXX84f/vAHPvOZz3DllVfS2tq6y/qWlhYWLFiwW7lzzhUyT5Yuq9555x3uuOMOADZv3kxdXd0u6+vr62lqatqt3DnnCllekqWkfSX9QtILklZIelxSlaQZkn6Xj5g6I2limGKsN9t+NeX1eyT9VtJzYXqzH0gaEdbNlnRLNmLOlvh1ys5Ow7a2trJhw4ZdyuIz2SxZsqRjFNnS0sLSpUsxM5YtW+ajS+fcgJHzZClJwG+AZWZ2iJkdD3wKeE+uY8mBjmQZ9rsO+I2ZHUY0J+ZYCvj5tPEEGZ+6K666upra2tpO22hvb+8YRdbX13ck0kQi4aNL59yAkY+R5QeBd8zs1mSBmb1kZjfHK0m6RtIVsdfrJE0My+dKWiNptaSfhbKDJC0O5YslHRjKzw7brpa0LJQNk3SDpOWh/he6E3gYZf5R0srw7/2hfD9JyyStCn19QNJ1wKhQdnfY721mdkfY553A5cAcSaNDFxMkPSSpSdLXQ9tjJD0Q4l8n6ZxOYpsrqVFSY3Nzc3d2J2caGho6fra3twNREk2WO+dcoctHsjwKWNnbjSUdBVQDHzSz44B/C6tuAe4ys2OBu4GbQvl84LRQ94xQdj7QZmZTganABZLSPxR1V68DHzazKcA5sT4+AzxsZpOB44BVZnYVsNXMJpvZrLDfK+KNmdkW4H+Bfw5FJwCzgMnA2ZIqgI8Ar5rZcWZ2NPBQusDMbKGZVZhZxfjx47uxK5nF57e88sor09apqalhn3326bKdysrKjp/FxdG3lYqLizvKnXOu0OX9Bh9JtWHUtLybm3wQuNfM3gAws7+H8hOBX4TlnwEnheUGYJGkC4BhoexU4FxJq4AngHHAYd3oezjwE0lrgV8DR4by5cDnJV0DHGNm/0izrQDLUP4/ZvY3M9tKdMr2JGAtcIqk6yV9wMxy9t2M+PclUyeGTiorK2POnDmdtlFcXMzMmTMBqKqqIjobHbWdLHfOuUKXj2S5HpiSfGFm84APAanDoXZ2jW9k+NlZ0kllof0LgauBCcAqSeNCG5eEUd9kMzvYzB7pRpuXA38lGj1WACNCH8uAacArwM8knZtm2/Vhmw6SSkJcG+Mxx/fBzJ4FjidKmt+SNL8bcebUhAkTdnmdTIgAM2bMoKysDIDy8nKmT5+OJKZNm9ZR7pxzhS4fyfJRYKSki2Jlo9PU20RIqpKmAMnTpIuBfw1JD0l7hfI/E90oBNGpzD+F9Yea2RNmNh94gyg5PQxcJGl4qHO4pDHdiL0UeM3MEsDnCCNVSQcBr5vZT4Cf8u6HgR3JPkLco5OJVNIw4DvAIjN7O9T5sKS9JI0CzgQaJO0PvG1mPwdujLVdkEaMGMHnP/95IEqiqaPHqqoqJk2a5KNK59yAkvPH3ZmZSToT+J6kLwPNwFtA6kWx+3j3VOly4Nmw/XpJNcBSSTuBp4DZwKXA7ZK+FNr8fGjnBkmHEY0mFwOrgTXARGBluEu1mSg5ZfJD4D5JZwOPhbgBZgBfkrQDeBNIjiwXAmskrTSzWZKqgB9K+hrRB5UHid0xS5Tgf0Z0DfMXZtYo6bSwDwlgBxD/kNGvSktLO+507epxd/F6ycfdnXLKKWnrlpeXM39+wQ2OnXOuS4p/J84NDhUVFdbY2JjVNltaWpg3b95u5T6fpXNusJC0wswq0q3L+w0+zjnnXKHzWUcCSccQnQKN225m78tHPM455wqHJ8vAzNYSfb/ROeec24UnS9ctpaWl3HzzzR2PwCspKaGoqIjS0tI8R+acc/3Pk6XrlqKiIsaNG8e4cePyHcqQE59cuz/7SP0glGs+cbgrZJ4snStwycm1Bzu/s9oVMv8Y55xzzmXgI0vnBpALz3ofY0ePyHq7b769nVvvfTL0cQJjR++R9T7S9/sOt977RE76cq4vPFm6ASF+3W4oX9saO3oEJWNGZq7Ypz726Pc+XPb4/43c8KPqBoTkdbt58+b1+80uzg0k/n8jNzxZOueccxl4snTOuUGkpaWF+fPnM3/+fFpbW9OuX7BgQdp1haAv8fXnvvUqWUp6M03ZhZ3M49jvJE2WdHo36lVIuqmXfey2z93c7jJJo2OvSyXdJWlj+HeXpNKwboak3/WmH+ecA6ivr+f555/n+eefp66uLu36pqamtOsKQV/i6899y9oNPmZ2a7baSkfSMDPb2cnqyUQTKz/YVRtm1ghkdzqOzC4Dfg4k56z8KbDOzJLzWn4DuA04O8dxDSjxKcIK9RNxf4nvbyIxuGYJiu/PUHtfsyV+3FpaWliyZEnH66VLlzJz5syOidZbWlpYunQpZsayZct2WVcI+hJff+9b1pKlpGuAN83sRklLgFXACUAJMMfMngwTNd8OHEKUPOaa2Zqw7aHAAUSTM3/bzH4iaQbwdeA1YHKYBPpHRImxHfh3oAFYAIySdBLwLeB3wM3AMWEfrzGz34b2rjCzj4c+DwyxHAh838wyjjoljQV+C5QDw4GrQ9tjgP8C3kM0KfQ3gX2B/YHHJL0BXAAcD5wTa3IB8LykQ8PrEkn1wCRgGfBFork4fxr224Dbzex7mWIdTJJPlwGorq7OYyT59fa2dyjbc1S+w8iat7e907E8lN/XbLn//vvZufPdMUV7ezt1dXXMmTMHiEZeyWkZE4nELusKQV/i6+99689rlmPM7P1Ef+xvD2XfAJ4ys2OJJj2+K1b/WOBjwInAfEn7h/ITgGozOxKYB2BmxwCfBu4M+zAfuMfMJpvZPUA18KiZTQVOJpo8eUyaGI8ATgt9fF3S8G7s1zagysymhLa/EyaQ/gjwqpkdZ2ZHAw+F5PsqcLKZnQwcCayKj5DD8irgqNj+/gdRoj8UmEk0cj7AzI4O+35HalCS5kpqlNTY3Nzcjd1wzg02q1ev7kgYAGZGQ0NDx+uGhgba29uBKJHG1xWCvsTX3/vWn9+z/CWAmS2TVCKpDDgJ+GQof1TSuOT1OuC3ZrYV2CrpMaKk0Qo8aWYvhjonEY0YMbNnJL0EHJ6m71OBMyRdEV6PJBo9pnrAzLYD2yW9TjQSfDnDfgm4VtI0IEE0Gt4XWAvcKOl64Hdm9sdOtk13Hi1e/qSZvQAg6ZdhnxcDh0i6GXgAeCS1ATNbCCyEaPLnDPsw4JSUlHQs19TUFNSpo/7W2traMeoaPTL7DyTIp/j+DLX3NVvivx9Tp06loaGhI2FKorKysqNuZWUlS5Ysob29neLi4l3WFYK+xNff+9afyTL1D7YRJYXO6qWrD/BWrCzd9ukI+KSZNe1SKO2bUm97bHkn3Tses4DxwPFmtkPSJmCkmT0r6XjgdOBbkh4xswUp264H3iupyMwSIaYi4DjgaaJTuLsdBzNrkXQc0Sh4HvCvQOGcO8mB+Bety8rKhuwzRIuKuvtfYGCI789Qfl+z5fTTT+cvf/lLxwiruLiYmTNndqyvqqpi6dKlQPR/Kr6uEPQlvv7et/48DXsOQLiO2GZmbUTX4GaF8hnAG2aWvBj1CUkjJY0DZgDL07QZ3/5wotFiE/APYM9YvYeBS8LpUSS9N4v7VQq8HhLlycBBoY/9gbfN7OfAjcCUUL8jNjN7HngKuDrW3tXAyrAO4ARJB4ckeg7wJ0l7A0Vmdh/wtVjbzjnXobS0lBkzZnS8nj59+i6j9fLycqZPn44kpk2bVnAj+b7E19/71tuR5WhJ8dOV301Tp0XSnwk3+ISya4A7JK0husHnvFj9J4lOMR4IfNPMXg0JMe6HwK2S1hLd4DPbzLaH07ZXSVpFdIPPN4HvA2tCwtwEfLyX+5rqbuC/JTUSXWt8JpQfQ3RtNAHsAC4K5QuB30t6LVy3PB+4WdLzRCPgx0NZ0uPAdaG9ZUB9WL4jJFCAr2RpX5xzg0xVVRWbNm0CSDu6qqqq4uWXXy64UWVSX+Lrz31T/GJw1hqN7oa9InxVozv1ryHcSZv1YIagiooKa2zM9Tdk+tdQfv5lS0tLxxRdV5z7gX55buuWt7Zx411/7Nc+MvXrU3T1zlD+v5FtklaYWUW6df4gdTcgFBUV+R9Solk6+qfd7WmX+1t/7c9Q4v83cqNfkqWZzehh/Wv6I46eCtdLF6dZ9SEz+1uu43EuVS6ms0pO1eWce5ePLGNCQpyc7zicc84VFk+WzhW40tJSamtr+7WPRCLR8ZSkkpKSvFz3Ki0tzVzJuTzxZOlcgcvVNalx48b1ex/ODVSeLJ1zQ1L8LtKebpeNUbjfuTqweLJ0zg1JbW1tHV/JyQf/qszA4h9rnHPOuQx8ZOmcG/I+8pGPMHJk9x7EsHXrVh5++GEATjvtNEaN6v6Uadu2beOhhx7qVYwuvzxZOudyqhCfODNy5EhGjx7d4+1GjRrVq+0KSSG+H4XIj4pzLqeS1wrnzZvXqxtsXHb5+9E9niydc73W0tLCV7/6VebMmcPatWtZsGABra2t+Q7L5VhLS8ugf+89WTrneq2+vp5Nmzaxbds2brrpJpqamqirq8t3WC7H6uvrB/17P2iuWUqqBj5DNIlzAvgCcD2wH7AV2AP4npktDPU3Ec01mQD+CpxrZv+vk7Y3ARVm9kYPY5oNPGJmr4bXI4BvA/839LsBmGdmL0uaCPzOzI7uSR/O5UtLSwtLlizpeP3WW9E87cuWLWPmzJmdzieYSCQ6lvM5Eon33R+zL6UT76dQRmHxOOLvTXe1tLSwdOlSzCzjez+QDYpkKelEovkqp4T5LfcGRoTVs8ysUdJewEZJi8wsOdXByWb2hqRrga8Cl2Y5tNnAOuDV8PpaoomgDzeznZI+D9RJel+W+3Wu39XX19Pe3r5beSKRoK6ujjlz5qTZio4v9ANUV1f3W3w9sW3bNsaMGZOTfpIKZd/jtmzZ0uMnOdXX13d8CMj03g9kg+U07H7AG2a2HcDM3kiO5mLGAm8RjTxTLQP+uTsdSfqNpBWS1kuaG8qGSVokaZ2ktZIul3QWUAHcLWmVpDHA54HLzWxniPMOYDvwwdB8saQ7Ja2RdK+k0aH96yRtCOVp5/yUNFdSo6TG5ubm7uyKc33S0NCQtry9vb3TdW7waWho6PjQNJjf+0ExsgQeAeZLehb4A3CPmS0N6+6WtB04DLgsmahSfBxY282+5pjZ3yWNApZLug+YCByQPIUqqczMWiVdTJgEW9KxwP+a2ZaU9hqBo4CNwCTgfDNrkHQ78MXwswo4wsxMUlm6oMLp5YUQTf7czX1xrtcqKyv5wx/+sFt5cXExlZWVnW5XUlLSsVxTU5O3U3atra0do7vufseyr+L95HPf4+LHIf7edFdlZSVLliyhvb0943s/kA2KZGlmb0o6HvgAcDJwj6SrwurkadjxwJ8lPWRmL4V1j0naCawBru5md5dKqgrLE4iScBNwiKSbgQeIkncqAemSWLx8s5klP5b9nOi08PeBbcBtkh4AftfNOJ3rV1VVVR1/JOOKioqYOXNmp9vFv8dXVlZWEI98k5Tzfgpl3+N68x3Lqqoqli5d2rF9V+/9QDZYTsNiZjvNbImZfR24GPhkyvpmYCUQvz54splNNrNzzaw1Ux+SZgCnACea2XHAU8BIM2sBjgOWAPOA29Js/jxwkKQ9U8qnEN3oA7snUzOzduAE4D7gTMAf/+EKQnl5OTNmzOh4PWbMGCQxbdq0ghgxudwoLy9n+vTpg/69HxTJUtIkSYfFiiYDL6XUGQ28l+h0Z2+VAi1m9rakI4B/CW3vDRSZ2X3A14gSIER32+4JYGZvAXcC35U0LGx3LjAaeDTUPzDcrATwaeBPksYCpWb2IHAZPjm1KyBVVVVMnDiRkSNHcumllzJp0qRBO7Jwnauqqhr07/2gOA1LdPPOzeF6XjvRKG4ucC/RNcvkV0cWmdmKPvTzEHChpDVEp17/EsoPAO6QlPzw8ZXwcxFwa+j/xFB+I/CspATwDFAVrkUCPA2cJ+nHwHPAj4gS9G8ljSQ6ZXt5H+J3LqvKy8u59tprO14fc8wxeYzG5Ut5eTnz58/Pdxj9alAky5AA359m1YwutpnYg/bjdT/aSbUpqQVhpHlfSvEl4V9q3U3AkWnafZvoNKxzg0JpaSm1tbUdyy6//P3onkGRLJ1zA0dRUVHB3dgS//5jJlu3bk27nO1+cqUQ349C5MkyRtITRKdr4z5nZt39WolzbgDq7bRZyam63ODnyTLGzPxJOs4553bjydI5NyTFr9X1RCKR6HhkX0lJSa/nf/TrgwOLJ0uXdcnJZNP9UfHJZV2h6Mu1up4+P9UNfJ4sXdYlJ5NNp7a21m8mcM4NOP4R3znnnMvAk6VzzjmXgSdL55xzLgO/Zun6JHkzD9Cjm3d6u51zzuWD/4VyfZK8mWfevHkdya8/t3POuXzwZOmcc85l4MnSOeecy2BAJ0tJb6a8ni3pltjrz0paI2m9pNWSbgvTeCFpiaQmSaskPS1pboa+NoV5K3sa42xJ+8dej5D0fUkbJT0n6beS3hPWTZS0rqd95FMikehYbm1tpaWlhdbW1k7rp6sTb8M55wrRoL3BR9JHiOZ+/KiZvRImXD4P2BdoDdVmmVmjpL2AjZIWmdk7WQ5lNrAOeDW8vpZoQujDzWynpM8DdZIG5HNpk0/oAaiurs5YP12dLVu2+BNRnHMFbUCPLDOoBq4ws1cAzGynmd1uZk1p6o4F3gJ2dqdhSb+RtCKMWOeGsmGSFklaJ2mtpMslnQVUEE1AvUrSGODzwOVmtjPEdQewHfhgaL5Y0p1hRHyvpNGh/eskbQjlN6aJaa6kRkmNzc3NPThMzjnnMhnoI8tRklbFXu8F3B+WjwJWZtj+bknbgcOAy5IJrBvmmNnfJY0Clku6D5gIHGBmRwNIKjOzVkkXEyXtRknHAv9rZltS2msM8W4EJgHnm1mDpNuBL4afVcARZmbJU8lxZrYQWAhQUVFh3dyPPispKelYrqmpoaysjNbW1k5HmenqxNtwzrlCNNBHllvNbHLyHzA/XSVJx4SR3UZJ58RWzTKzY4EDgSskHdTNfi+VtBr4CzCBKNm+ABwi6eZwCjg1IQIISJfI4uWbzawhLP8cOCm0tQ24TdJM4O1uxtnv4t+PLCsro7y8nLKysk7rp6vj37F0zhW6wfxXaj0wBcDM1oZk+ntgVGpFM2smGoVmvG4oaQZwCnCimR0HPAWMNLMW4DhgCTAPuC3N5s8DB0naM6V8CrAhGc7u4Vk7cAJwH3Am0LuZap1zzvXKYE6W3wJuTN5pGuyWKAHCdcH3Ep0GzaQUaDGztyUdAfxLaGNvoMjM7gO+RkjUwD+IbujBzN4C7gS+G244QtK5wGjg0VD/QEknhuVPA3+SNBYoNbMHgcuAyd2I0znnXJYM9GuWnTKzByWNB34fElMr0V2pD8eq3S1pK7AHsMjMVnSj6YeACyWtAZqITsUCHADcISn5AeQr4eci4NbQz4mh/EbgWUkJ4BmgKlyLBHgaOE/Sj4HngB8RJejfShpJdMr28h4dDOecc30is5zdC+JypKKiwhobG3PSV7pnvLa0tGScz9KfDeucKzSSVphZRbp1g3Zk6XKjt7PN92WWeuecyzVPlikkPUF0Wjbuc2a2Nh/xOOecyz9PlinMbEA+Scc551z/8WTpsq60tJTa2loSiUTH4/BKSkooKiqitLQ0z9E551zPebJ0Hdrb29m8eTMQJbfy8vJe3XgTvx7pz3x1zg0Gnixdh82bN+/ymLrknavOOTfU+f36zjnnXAaeLJ1zzrkMPFk655xzGXiydAC73Lma1NraSiKRyFNEzjlXODxZOgDa2tq4/vrrdymrrq7ueCSdc84NZZ4sXZdqampobW3NdxjOOZdXnixdl1599VXq6uryHYZzzuVV1pKlpH+S9CtJGyVtkPSgpMP72GaZpC/GXu8v6d5ubPegpLK+9J3S3jWSrujFdpMlnZ5SdqakNZKekbRW0pmxdUskpX3ifX/r6trkY4895qNL59yQlpVkqWgixnpgiZkdamZHAl8F9o3VGdaLpsuAjmRpZq+a2VmZNjKz082stRf9ZdtkoCNZSjqOaC7LT5jZEcAZRBNUH5uf8N6VenNP3M6dO3106Zwb0rI1sjwZ2GFmtyYLzGwVMEzSY5J+AayVNEzSDZKWh9HVFwAkjZW0WNLKMNr6RGjmOuBQSavCdhMlrQvbzJZUJ+khSc9J+nayb0mbJO0d6j8t6SeS1kt6RNKoUOfQsO0KSX+UdER3dlTSBSH+1ZLukzQ6lJ8taV0oXyZpBLAAOCfEfw5wBXCtmb0YjtGLwLeAL8W6+KykP4e2TghtTw9trJL0lKQ908Q1V1KjpMbm5ubu7EqPNDQ0ZL1N55wbKLKVLI8GVnSy7gSgOow2zwfazGwqMBW4QNLBwDagysymECXe74TR6lXARjObbGZfStP2ZOAc4BiipDQhTZ3DgFozOwpoBT4ZyhcCl5jZ8URJ7Ifd3Nc6M5tqZscBT4d9ApgPnBbKzzCzd0LZPSH+e4Cj2P04NYbypDFm9n6iEfXtoewKYJ6ZTQY+AGxNDcrMFppZhZlVjB8/vpu78q6SkpIu11dWVva4TeecGyxy8WzYJ5MjKeBU4FhJyVOppUTJ7GXgWknTgARwALFTuF1YbGZtAJI2AAcBm1PqvBhGuRAlqomSxgLvB34d5WRg9zksO3O0pP8kOkU8Fng4lDcAiyT9F9DZOUsBlqHslwBmtkxSSbj22gB8V9LdRMn65W7G2m1dPTB92LBhzJw5M9tdOufcgJGtZLke6Oxa4luxZRGN5h6OV5A0GxgPHG9mOyRtAkZ2o9/tseWdpN+f1DqjiEbUrWGk1lOLgDPNbHWIewaAmV0o6X3Ax4BVktK1vR6oANbEyqYAG2KvU5Opmdl1kh4guv75F0mnmNkzvYi9V04++WTKyspy1Z1zzhWcbJ2GfRTYQ9IFyQJJU4HpKfUeBi6SNDzUOVzSGKIR5ushUZ5MNEIE+Aew2/W5vjKzLcCLks4OcSjcfNMdewKvhX2YlSyUdKiZPWFm84E3gAnsHv+NwFckTQzbTCS6Eeo7sTrnhHUnEZ2ybgttrzWz64lO23br+mo27L///j6qdM4NeVlJlmZmQBXw4fDVkfXANcCrKVVvIxpFrQw36vyYaDR4N1AhqZEoAT0T2v0b0BBudrkhG7HGzALOl7SaaMT3iQz1k74GPAH8TzLO4IZwc9I6YBmwGngMODJ5g084HXwl8N+SngH+G/hy7DQxQIukPwO38u710MuSNw8RXa/8fc93t2ulpaVceeWVu5TV1NTw7W9/20eVzrkhT1Gec4NJRUWFNTY29ni7F1980eezdM4NWZJWmFna77r7E3ycc865DHJxN+yAIakaODul+NdmVpOPeJxzzhUGT5YxISl6YnTOObcLT5auw4QJE6ipiT4rlJSUUFpamueInHOuMPgNPoOQpGbgpV5sujfR114KUaHGVqhxQeHGVqhxQeHGVqhxQeHG1pu4DjKztI9A82TpOkhq7OxOsHwr1NgKNS4o3NgKNS4o3NgKNS4o3NiyHZffDeucc85l4MnSOeecy8CTpYtbmO8AulCosRVqXFC4sRVqXFC4sRVqXFC4sWU1Lr9m6ZxzzmXgI0vnnHMuA0+WzjnnXAaeLB0Akj4iqUnS85KuymMcEyQ9JulpSesl/Vsov0bSK2EGl1WSTs9TfJvC7DKrwiw5SNpL0v9Iei78zOnT5yVNih2XVZK2SLosX8dM0u2SXg8z8CTLOj1Gkr4Sfu+aJJ2W47hukPSMpDWS6sNk60iaKGlr7Njd2l9xdRFbp+9fno/ZPbGYNklaFcpzdsy6+DvRf79nZub/hvg/YBiwETgEGEE0vdiReYplP2BKWN4TeBY4kmjKtysK4FhtAvZOKfs2cFVYvgq4Ps/v5f8jmhM2L8cMmEY0qfm6TMcovLergT2Ag8Pv4bAcxnUqUByWr4/FNTFeL0/HLO37l+9jlrL+O8D8XB+zLv5O9NvvmY8sHcAJwPNm9oKZvQP8iu7P75lVZvaama0My/8AngYOyEcsPfAJ4M6wfCdwZv5C4UPARjPrzROcssLMlgF/Tynu7Bh9AviVmW03sxeB54l+H3MSl5k9Ymbt4eVfgPf0R9+ZdHLMOpPXY5YkScC/Ar/sj7670sXfiX77PfNk6SD6Jdsce/0yBZCgJE0E3ks02TbAxeF02e25PtUZY8AjklZImhvK9jWz1yD6Twzsk6fYAD7Frn+8CuGYQefHqJB+9+aw68TqB0t6StJSSR/IU0zp3r9COWYfAP5qZs/FynJ+zFL+TvTb75knSwegNGV5/U6RpLHAfcBlZrYF+BFwKDAZeI3o9E8+VJrZFOCjwDxJ0/IUx24kjQDOAH4digrlmHWlIH73FE3P1w7cHYpeAw40s/cC/w78QlJJjsPq7P0riGMGfJpdP5jl/Jil+TvRadU0ZT06Zp4sHUSfsibEXr8HeDVPsSBpONF/gLvNrA7AzP5qZjvNLAH8hH467ZSJmb0afr4O1Ic4/ippvxD7fsDr+YiNKIGvNLO/hhgL4pgFnR2jvP/uSToP+Dgwy8IFrnC67m9heQXRNa7DcxlXF+9fIRyzYmAmcE+yLNfHLN3fCfrx98yTpQNYDhwm6eAwOvkUcH8+AgnXQX4KPG1m342V7xerVgWsS902B7GNkbRncpno5pB1RMfqvFDtPOC3uY4t2OWTfiEcs5jOjtH9wKck7SHpYOAw4MlcBSXpI8CVwBlm9nasfLykYWH5kBDXC7mKK/Tb2fuX12MWnAI8Y2YvJwtyecw6+ztBf/6e5eLOJf9X+P+A04nuKNsIVOcxjpOITo+sAVaFf6cDPwPWhvL7gf3yENshRHfUrQbWJ48TMA5YDDwXfu6Vh9hGA38DSmNleTlmRAn7NWAH0Sf687s6RkB1+L1rAj6a47ieJ7qWlfxduzXU/WR4j1cDK4H/m4dj1un7l89jFsoXARem1M3ZMevi70S//Z754+6cc865DPw0rHPOOZeBJ0vnnHMuA0+WzjnnXAaeLJ1zzrkMPFk655xzGXiydM455zLwZOmcc85l8P8B/6SOC0jcsYsAAAAASUVORK5CYII=\n",
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
    "outlier_pos,dframe_outlierRemoved=my_internal_func.my_outlier_flagbyZscore(dframe_final[predictors])"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8af3e502",
   "metadata": {},
   "source": [
    "#### <font color='green'> Class Imbalance</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "20887feb",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "   target   id\n",
      "0       0  200\n",
      "1       1  100\n"
     ]
    }
   ],
   "source": [
    "print(dframe_final.groupby(['target'],as_index=False)['id'].nunique())\n",
    "\n",
    "# Imbalance of data is handled by assigning class weights to the models.\n",
    "# Oversampling is avoided as it will not add any information value\n",
    "idx_0=dframe_final[dframe_final['target']==0].sample(100,random_state=0).index\n",
    "idx_1=dframe_final[dframe_final['target']==1].index\n",
    "\n",
    "dframe_outlierremoved_balanced=dframe_final[~dframe_final.index.isin(outlier_pos)]"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "c864d6d0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "id                                           int64\n",
      "race                                        object\n",
      "gender                                      object\n",
      "Age Group                                   object\n",
      "target                                       int64\n",
      "time                                         int64\n",
      "CKD(t=0)                                    object\n",
      "Dieabetes(t=0)                              object\n",
      "Cholestrol(t=0)                             object\n",
      "Hyper Tension(t=0)                          object\n",
      "Hemoglobin(t=0)                             object\n",
      "Creatinine_baseline                        float64\n",
      "Creatinine_lastObs                         float64\n",
      "SBP_baseline                               float64\n",
      "SBP_lastObs                                float64\n",
      "HGB_baseline                               float64\n",
      "HGB_lastObs                                float64\n",
      "Glucose_baseline                           float64\n",
      "Glucose_lastObs                            float64\n",
      "Lipoprotein_baseline                       float64\n",
      "Lipoprotein_lastObs                        float64\n",
      "Medication Period-Diabetes Type 2           object\n",
      "Medication Period-High Blood Cholestrol     object\n",
      "Medication Period-Hyper Tension             object\n",
      "dtype: object\n",
      "Your selected dataframe has 24 columns.\n",
      "There are 0 columns that have missing values.\n",
      "(300, 22)\n"
     ]
    }
   ],
   "source": [
    "# Set Index\n",
    "my_internal_func.my_missing_values_table(dframe_outlierremoved_balanced)\n",
    "del(dframe_outlierremoved_balanced['time'])\n",
    "dframe_outlierremoved_balanced=dframe_outlierremoved_balanced.set_index('id')\n",
    "print(dframe_outlierremoved_balanced.shape)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "594e62a4",
   "metadata": {},
   "source": [
    "#### <font color='green'> Scaling of Data</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "1b5f0be4",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Original shape of dataframe-----> (300, 22)\n",
      "race                                        object\n",
      "gender                                      object\n",
      "Age Group                                   object\n",
      "target                                       int64\n",
      "CKD(t=0)                                    object\n",
      "Dieabetes(t=0)                              object\n",
      "Cholestrol(t=0)                             object\n",
      "Hyper Tension(t=0)                          object\n",
      "Hemoglobin(t=0)                             object\n",
      "Creatinine_baseline                        float64\n",
      "Creatinine_lastObs                         float64\n",
      "SBP_baseline                               float64\n",
      "SBP_lastObs                                float64\n",
      "HGB_baseline                               float64\n",
      "HGB_lastObs                                float64\n",
      "Glucose_baseline                           float64\n",
      "Glucose_lastObs                            float64\n",
      "Lipoprotein_baseline                       float64\n",
      "Lipoprotein_lastObs                        float64\n",
      "Medication Period-Diabetes Type 2           object\n",
      "Medication Period-High Blood Cholestrol     object\n",
      "Medication Period-Hyper Tension             object\n",
      "dtype: object\n",
      "Your selected dataframe has 22 columns.\n",
      "There are 0 columns that have missing values.\n",
      "\n",
      " ............................  Initiating feature scaling.....\n",
      "\n",
      "\n",
      "Columns that are numerics----> 10\n",
      "\n",
      " --------------- Scaling of features completed with new dataframe size---------> (300, 10)\n",
      "\n",
      "--------------------Initiating categorical encoding--------------------------------\n",
      "\n",
      "\n",
      "The columns to be encoded------- Index(['Age Group', 'CKD(t=0)', 'Cholestrol(t=0)', 'Dieabetes(t=0)',\n",
      "       'Hemoglobin(t=0)', 'Hyper Tension(t=0)',\n",
      "       'Medication Period-Diabetes Type 2',\n",
      "       'Medication Period-High Blood Cholestrol',\n",
      "       'Medication Period-Hyper Tension', 'gender', 'race'],\n",
      "      dtype='object')\n",
      "\n",
      "------------Completed Scaling and Encoding-----------------------\n",
      "\n",
      "\n",
      " Final dataframe size\n",
      " (300, 22)\n",
      "Creatinine_baseline                        float64\n",
      "Creatinine_lastObs                         float64\n",
      "Glucose_baseline                           float64\n",
      "Glucose_lastObs                            float64\n",
      "HGB_baseline                               float64\n",
      "HGB_lastObs                                float64\n",
      "Lipoprotein_baseline                       float64\n",
      "Lipoprotein_lastObs                        float64\n",
      "SBP_baseline                               float64\n",
      "SBP_lastObs                                float64\n",
      "Age Group                                    int32\n",
      "CKD(t=0)                                     int32\n",
      "Cholestrol(t=0)                              int32\n",
      "Dieabetes(t=0)                               int32\n",
      "Hemoglobin(t=0)                              int32\n",
      "Hyper Tension(t=0)                           int32\n",
      "Medication Period-Diabetes Type 2            int32\n",
      "Medication Period-High Blood Cholestrol      int32\n",
      "Medication Period-Hyper Tension              int32\n",
      "gender                                       int32\n",
      "race                                         int32\n",
      "target                                       int64\n",
      "dtype: object\n",
      "Your selected dataframe has 22 columns.\n",
      "There are 0 columns that have missing values.\n"
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
       "      <th>Creatinine_baseline</th>\n",
       "      <th>Creatinine_lastObs</th>\n",
       "      <th>Glucose_baseline</th>\n",
       "      <th>Glucose_lastObs</th>\n",
       "      <th>HGB_baseline</th>\n",
       "      <th>HGB_lastObs</th>\n",
       "      <th>Lipoprotein_baseline</th>\n",
       "      <th>Lipoprotein_lastObs</th>\n",
       "      <th>SBP_baseline</th>\n",
       "      <th>SBP_lastObs</th>\n",
       "      <th>...</th>\n",
       "      <th>Cholestrol(t=0)</th>\n",
       "      <th>Dieabetes(t=0)</th>\n",
       "      <th>Hemoglobin(t=0)</th>\n",
       "      <th>Hyper Tension(t=0)</th>\n",
       "      <th>Medication Period-Diabetes Type 2</th>\n",
       "      <th>Medication Period-High Blood Cholestrol</th>\n",
       "      <th>Medication Period-Hyper Tension</th>\n",
       "      <th>gender</th>\n",
       "      <th>race</th>\n",
       "      <th>target</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>-0.026022</td>\n",
       "      <td>-0.389869</td>\n",
       "      <td>-0.392190</td>\n",
       "      <td>-0.551095</td>\n",
       "      <td>-0.375755</td>\n",
       "      <td>-0.430711</td>\n",
       "      <td>2.527166</td>\n",
       "      <td>2.551382</td>\n",
       "      <td>-0.020086</td>\n",
       "      <td>1.000331</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1.130516</td>\n",
       "      <td>2.421685</td>\n",
       "      <td>2.046961</td>\n",
       "      <td>1.974931</td>\n",
       "      <td>-0.119121</td>\n",
       "      <td>-0.605167</td>\n",
       "      <td>-0.021326</td>\n",
       "      <td>-0.454071</td>\n",
       "      <td>1.604377</td>\n",
       "      <td>1.150067</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>3.443592</td>\n",
       "      <td>2.421685</td>\n",
       "      <td>0.249692</td>\n",
       "      <td>0.206713</td>\n",
       "      <td>0.779098</td>\n",
       "      <td>0.790478</td>\n",
       "      <td>-1.013785</td>\n",
       "      <td>0.039150</td>\n",
       "      <td>-0.103392</td>\n",
       "      <td>0.251648</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>-1.760829</td>\n",
       "      <td>-1.233335</td>\n",
       "      <td>-0.263813</td>\n",
       "      <td>-0.551095</td>\n",
       "      <td>-0.247438</td>\n",
       "      <td>-0.663319</td>\n",
       "      <td>0.336668</td>\n",
       "      <td>0.546564</td>\n",
       "      <td>0.910162</td>\n",
       "      <td>0.687245</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>-0.026022</td>\n",
       "      <td>-1.233335</td>\n",
       "      <td>1.276703</td>\n",
       "      <td>2.732739</td>\n",
       "      <td>-0.568230</td>\n",
       "      <td>-1.302989</td>\n",
       "      <td>-0.868461</td>\n",
       "      <td>-0.496651</td>\n",
       "      <td>-0.388020</td>\n",
       "      <td>2.286704</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
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
       "      <td>1.708785</td>\n",
       "      <td>0.734753</td>\n",
       "      <td>-0.199625</td>\n",
       "      <td>-0.551095</td>\n",
       "      <td>-0.247438</td>\n",
       "      <td>0.267111</td>\n",
       "      <td>0.762008</td>\n",
       "      <td>1.341395</td>\n",
       "      <td>1.167022</td>\n",
       "      <td>0.135943</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>296</th>\n",
       "      <td>1.708785</td>\n",
       "      <td>0.172442</td>\n",
       "      <td>0.249692</td>\n",
       "      <td>0.775069</td>\n",
       "      <td>0.714939</td>\n",
       "      <td>0.848629</td>\n",
       "      <td>0.361479</td>\n",
       "      <td>0.205923</td>\n",
       "      <td>0.368674</td>\n",
       "      <td>0.707664</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>4</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>297</th>\n",
       "      <td>-0.315157</td>\n",
       "      <td>-0.108713</td>\n",
       "      <td>0.955762</td>\n",
       "      <td>1.848630</td>\n",
       "      <td>-0.054962</td>\n",
       "      <td>0.325263</td>\n",
       "      <td>1.587875</td>\n",
       "      <td>2.345578</td>\n",
       "      <td>0.778261</td>\n",
       "      <td>0.272067</td>\n",
       "      <td>...</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>3</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>298</th>\n",
       "      <td>-0.893426</td>\n",
       "      <td>-0.952179</td>\n",
       "      <td>-0.263813</td>\n",
       "      <td>-0.424793</td>\n",
       "      <td>0.779098</td>\n",
       "      <td>1.953515</td>\n",
       "      <td>-0.393498</td>\n",
       "      <td>-0.847937</td>\n",
       "      <td>0.361732</td>\n",
       "      <td>0.040656</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>299</th>\n",
       "      <td>-0.604291</td>\n",
       "      <td>-1.514490</td>\n",
       "      <td>-0.777319</td>\n",
       "      <td>-0.298492</td>\n",
       "      <td>0.458305</td>\n",
       "      <td>0.383415</td>\n",
       "      <td>-0.843649</td>\n",
       "      <td>-0.248266</td>\n",
       "      <td>-2.109673</td>\n",
       "      <td>-1.320585</td>\n",
       "      <td>...</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>300 rows × 22 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Creatinine_baseline  Creatinine_lastObs  Glucose_baseline  \\\n",
       "0              -0.026022           -0.389869         -0.392190   \n",
       "1               1.130516            2.421685          2.046961   \n",
       "2               3.443592            2.421685          0.249692   \n",
       "3              -1.760829           -1.233335         -0.263813   \n",
       "4              -0.026022           -1.233335          1.276703   \n",
       "..                   ...                 ...               ...   \n",
       "295             1.708785            0.734753         -0.199625   \n",
       "296             1.708785            0.172442          0.249692   \n",
       "297            -0.315157           -0.108713          0.955762   \n",
       "298            -0.893426           -0.952179         -0.263813   \n",
       "299            -0.604291           -1.514490         -0.777319   \n",
       "\n",
       "     Glucose_lastObs  HGB_baseline  HGB_lastObs  Lipoprotein_baseline  \\\n",
       "0          -0.551095     -0.375755    -0.430711              2.527166   \n",
       "1           1.974931     -0.119121    -0.605167             -0.021326   \n",
       "2           0.206713      0.779098     0.790478             -1.013785   \n",
       "3          -0.551095     -0.247438    -0.663319              0.336668   \n",
       "4           2.732739     -0.568230    -1.302989             -0.868461   \n",
       "..               ...           ...          ...                   ...   \n",
       "295        -0.551095     -0.247438     0.267111              0.762008   \n",
       "296         0.775069      0.714939     0.848629              0.361479   \n",
       "297         1.848630     -0.054962     0.325263              1.587875   \n",
       "298        -0.424793      0.779098     1.953515             -0.393498   \n",
       "299        -0.298492      0.458305     0.383415             -0.843649   \n",
       "\n",
       "     Lipoprotein_lastObs  SBP_baseline  SBP_lastObs  ...  Cholestrol(t=0)  \\\n",
       "0               2.551382     -0.020086     1.000331  ...                1   \n",
       "1              -0.454071      1.604377     1.150067  ...                0   \n",
       "2               0.039150     -0.103392     0.251648  ...                0   \n",
       "3               0.546564      0.910162     0.687245  ...                0   \n",
       "4              -0.496651     -0.388020     2.286704  ...                0   \n",
       "..                   ...           ...          ...  ...              ...   \n",
       "295             1.341395      1.167022     0.135943  ...                1   \n",
       "296             0.205923      0.368674     0.707664  ...                1   \n",
       "297             2.345578      0.778261     0.272067  ...                1   \n",
       "298            -0.847937      0.361732     0.040656  ...                0   \n",
       "299            -0.248266     -2.109673    -1.320585  ...                0   \n",
       "\n",
       "     Dieabetes(t=0)  Hemoglobin(t=0)  Hyper Tension(t=0)  \\\n",
       "0                 0                0                   1   \n",
       "1                 1                0                   0   \n",
       "2                 1                0                   1   \n",
       "3                 0                0                   1   \n",
       "4                 1                0                   0   \n",
       "..              ...              ...                 ...   \n",
       "295               0                0                   1   \n",
       "296               1                0                   0   \n",
       "297               1                0                   0   \n",
       "298               0                0                   1   \n",
       "299               0                0                   0   \n",
       "\n",
       "     Medication Period-Diabetes Type 2  \\\n",
       "0                                    1   \n",
       "1                                    0   \n",
       "2                                    0   \n",
       "3                                    0   \n",
       "4                                    1   \n",
       "..                                 ...   \n",
       "295                                  0   \n",
       "296                                  1   \n",
       "297                                  1   \n",
       "298                                  1   \n",
       "299                                  0   \n",
       "\n",
       "     Medication Period-High Blood Cholestrol  Medication Period-Hyper Tension  \\\n",
       "0                                          1                                1   \n",
       "1                                          1                                0   \n",
       "2                                          1                                0   \n",
       "3                                          1                                0   \n",
       "4                                          1                                1   \n",
       "..                                       ...                              ...   \n",
       "295                                        0                                0   \n",
       "296                                        1                                0   \n",
       "297                                        1                                1   \n",
       "298                                        1                                0   \n",
       "299                                        1                                1   \n",
       "\n",
       "     gender  race  target  \n",
       "0         1     3       1  \n",
       "1         0     4       0  \n",
       "2         0     4       1  \n",
       "3         1     4       0  \n",
       "4         0     4       1  \n",
       "..      ...   ...     ...  \n",
       "295       0     4       1  \n",
       "296       0     4       0  \n",
       "297       1     3       1  \n",
       "298       0     0       0  \n",
       "299       1     0       0  \n",
       "\n",
       "[300 rows x 22 columns]"
      ]
     },
     "execution_count": 16,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Standard scaling on numerical data \n",
    "\n",
    "\n",
    "dframe_outlierremoved_balanced_scaled=my_internal_func.my_preprocess(dframe_outlierremoved_balanced,'standard',ignore=['target'])\n",
    "dframe_outlierremoved_balanced_scaled"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "7f6f393b",
   "metadata": {},
   "source": [
    "#### <font color='green'> Inspect weight of Evidence and Information Value</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "f9c1c4df",
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
       "      <th>VAR_NAME</th>\n",
       "      <th>IV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>CKD(t=0)</td>\n",
       "      <td>0.366782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>HGB_baseline</td>\n",
       "      <td>0.204297</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>gender</td>\n",
       "      <td>0.198980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Hemoglobin(t=0)</td>\n",
       "      <td>0.109971</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>SBP_baseline</td>\n",
       "      <td>0.073501</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Lipoprotein_baseline</td>\n",
       "      <td>0.058709</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Age Group</td>\n",
       "      <td>0.038612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>0.037664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Medication Period-Hyper Tension</td>\n",
       "      <td>0.029897</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>race</td>\n",
       "      <td>0.022637</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Dieabetes(t=0)</td>\n",
       "      <td>0.022056</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Hyper Tension(t=0)</td>\n",
       "      <td>0.018551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Cholestrol(t=0)</td>\n",
       "      <td>0.016581</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Glucose_baseline</td>\n",
       "      <td>0.003601</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Medication Period-High Blood Cholestrol</td>\n",
       "      <td>0.000556</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                   VAR_NAME        IV\n",
       "1                                  CKD(t=0)  0.366782\n",
       "5                              HGB_baseline  0.204297\n",
       "13                                   gender  0.198980\n",
       "6                           Hemoglobin(t=0)  0.109971\n",
       "12                             SBP_baseline  0.073501\n",
       "8                      Lipoprotein_baseline  0.058709\n",
       "0                                 Age Group  0.038612\n",
       "9         Medication Period-Diabetes Type 2  0.037664\n",
       "11          Medication Period-Hyper Tension  0.029897\n",
       "14                                     race  0.022637\n",
       "3                            Dieabetes(t=0)  0.022056\n",
       "7                        Hyper Tension(t=0)  0.018551\n",
       "2                           Cholestrol(t=0)  0.016581\n",
       "4                          Glucose_baseline  0.003601\n",
       "10  Medication Period-High Blood Cholestrol  0.000556"
      ]
     },
     "execution_count": 17,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# Inspect Information Value\n",
    "\n",
    "# print(dframe_outlierremoved_balanced.info())\n",
    "\n",
    "import MyInformationValue as my_iv\n",
    "\n",
    "featureset_first= ['race', 'gender', 'Age Group',\n",
    "                   'CKD(t=0)', 'Dieabetes(t=0)', 'Cholestrol(t=0)', 'Hyper Tension(t=0)', 'Hemoglobin(t=0)',\n",
    "                    'Glucose_baseline', 'Lipoprotein_baseline', 'SBP_baseline', 'HGB_baseline', \n",
    "                   'Medication Period-Diabetes Type 2', 'Medication Period-High Blood Cholestrol', 'Medication Period-Hyper Tension']\n",
    "\n",
    "\n",
    "final_iv,IV=my_iv.data_vars(dframe_outlierremoved_balanced[featureset_first],dframe_outlierremoved_balanced.target)\n",
    "final_iv.to_csv('output/InformationValue_1.csv')\n",
    "IV.sort_values('IV',ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "8fa0aa75",
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
       "      <th>VAR_NAME</th>\n",
       "      <th>IV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>CKD(t=0)</td>\n",
       "      <td>0.366782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>gender</td>\n",
       "      <td>0.198980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Glucose_lastObs</td>\n",
       "      <td>0.169468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>HGB_lastObs</td>\n",
       "      <td>0.164081</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Hemoglobin(t=0)</td>\n",
       "      <td>0.109971</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>Lipoprotein_lastObs</td>\n",
       "      <td>0.101544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Age Group</td>\n",
       "      <td>0.038612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>0.037664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>Medication Period-Hyper Tension</td>\n",
       "      <td>0.029897</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Dieabetes(t=0)</td>\n",
       "      <td>0.022056</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Hyper Tension(t=0)</td>\n",
       "      <td>0.018551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Cholestrol(t=0)</td>\n",
       "      <td>0.016581</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>Medication Period-High Blood Cholestrol</td>\n",
       "      <td>0.000556</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                   VAR_NAME        IV\n",
       "1                                  CKD(t=0)  0.366782\n",
       "13                                   gender  0.198980\n",
       "12                              SBP_lastObs  0.184565\n",
       "4                           Glucose_lastObs  0.169468\n",
       "5                               HGB_lastObs  0.164081\n",
       "6                           Hemoglobin(t=0)  0.109971\n",
       "8                       Lipoprotein_lastObs  0.101544\n",
       "0                                 Age Group  0.038612\n",
       "9         Medication Period-Diabetes Type 2  0.037664\n",
       "11          Medication Period-Hyper Tension  0.029897\n",
       "3                            Dieabetes(t=0)  0.022056\n",
       "7                        Hyper Tension(t=0)  0.018551\n",
       "2                           Cholestrol(t=0)  0.016581\n",
       "10  Medication Period-High Blood Cholestrol  0.000556"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "new_reduced_feature=['gender', 'Age Group',\n",
    "                   'CKD(t=0)', 'Dieabetes(t=0)', 'Cholestrol(t=0)', 'Hyper Tension(t=0)', 'Hemoglobin(t=0)',\n",
    "                    'Glucose_lastObs', 'Lipoprotein_lastObs', 'SBP_lastObs', 'HGB_lastObs', \n",
    "                     'Medication Period-Diabetes Type 2', 'Medication Period-High Blood Cholestrol', 'Medication Period-Hyper Tension'\n",
    "                   ]\n",
    "                     \n",
    "final_iv,IV=my_iv.data_vars(dframe_outlierremoved_balanced[new_reduced_feature],dframe_outlierremoved_balanced.target)\n",
    "IV.sort_values('IV',ascending=False)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "ed5d37ce",
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
       "      <th>VAR_NAME</th>\n",
       "      <th>MIN_VALUE</th>\n",
       "      <th>MAX_VALUE</th>\n",
       "      <th>COUNT</th>\n",
       "      <th>EVENT</th>\n",
       "      <th>EVENT_RATE</th>\n",
       "      <th>NONEVENT</th>\n",
       "      <th>NON_EVENT_RATE</th>\n",
       "      <th>DIST_EVENT</th>\n",
       "      <th>DIST_NON_EVENT</th>\n",
       "      <th>WOE</th>\n",
       "      <th>IV</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>gender</td>\n",
       "      <td>Female</td>\n",
       "      <td>Female</td>\n",
       "      <td>176</td>\n",
       "      <td>44</td>\n",
       "      <td>0.250000</td>\n",
       "      <td>132</td>\n",
       "      <td>0.750000</td>\n",
       "      <td>0.44</td>\n",
       "      <td>0.660</td>\n",
       "      <td>-0.405465</td>\n",
       "      <td>0.198980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>gender</td>\n",
       "      <td>Male</td>\n",
       "      <td>Male</td>\n",
       "      <td>124</td>\n",
       "      <td>56</td>\n",
       "      <td>0.451613</td>\n",
       "      <td>68</td>\n",
       "      <td>0.548387</td>\n",
       "      <td>0.56</td>\n",
       "      <td>0.340</td>\n",
       "      <td>0.498991</td>\n",
       "      <td>0.198980</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Age Group</td>\n",
       "      <td>40-50</td>\n",
       "      <td>40-50</td>\n",
       "      <td>10</td>\n",
       "      <td>4</td>\n",
       "      <td>0.400000</td>\n",
       "      <td>6</td>\n",
       "      <td>0.600000</td>\n",
       "      <td>0.04</td>\n",
       "      <td>0.030</td>\n",
       "      <td>0.287682</td>\n",
       "      <td>0.038612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Age Group</td>\n",
       "      <td>51-60</td>\n",
       "      <td>51-60</td>\n",
       "      <td>37</td>\n",
       "      <td>14</td>\n",
       "      <td>0.378378</td>\n",
       "      <td>23</td>\n",
       "      <td>0.621622</td>\n",
       "      <td>0.14</td>\n",
       "      <td>0.115</td>\n",
       "      <td>0.196710</td>\n",
       "      <td>0.038612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Age Group</td>\n",
       "      <td>61-70</td>\n",
       "      <td>61-70</td>\n",
       "      <td>93</td>\n",
       "      <td>35</td>\n",
       "      <td>0.376344</td>\n",
       "      <td>58</td>\n",
       "      <td>0.623656</td>\n",
       "      <td>0.35</td>\n",
       "      <td>0.290</td>\n",
       "      <td>0.188052</td>\n",
       "      <td>0.038612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Age Group</td>\n",
       "      <td>71-80</td>\n",
       "      <td>71-80</td>\n",
       "      <td>116</td>\n",
       "      <td>35</td>\n",
       "      <td>0.301724</td>\n",
       "      <td>81</td>\n",
       "      <td>0.698276</td>\n",
       "      <td>0.35</td>\n",
       "      <td>0.405</td>\n",
       "      <td>-0.145954</td>\n",
       "      <td>0.038612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>Age Group</td>\n",
       "      <td>81 and Above</td>\n",
       "      <td>81 and Above</td>\n",
       "      <td>44</td>\n",
       "      <td>12</td>\n",
       "      <td>0.272727</td>\n",
       "      <td>32</td>\n",
       "      <td>0.727273</td>\n",
       "      <td>0.12</td>\n",
       "      <td>0.160</td>\n",
       "      <td>-0.287682</td>\n",
       "      <td>0.038612</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>CKD(t=0)</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>17</td>\n",
       "      <td>8</td>\n",
       "      <td>0.470588</td>\n",
       "      <td>9</td>\n",
       "      <td>0.529412</td>\n",
       "      <td>0.08</td>\n",
       "      <td>0.045</td>\n",
       "      <td>0.575364</td>\n",
       "      <td>0.366782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>CKD(t=0)</td>\n",
       "      <td>2.0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>71</td>\n",
       "      <td>31</td>\n",
       "      <td>0.436620</td>\n",
       "      <td>40</td>\n",
       "      <td>0.563380</td>\n",
       "      <td>0.31</td>\n",
       "      <td>0.200</td>\n",
       "      <td>0.438255</td>\n",
       "      <td>0.366782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>CKD(t=0)</td>\n",
       "      <td>3.0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>85</td>\n",
       "      <td>32</td>\n",
       "      <td>0.376471</td>\n",
       "      <td>53</td>\n",
       "      <td>0.623529</td>\n",
       "      <td>0.32</td>\n",
       "      <td>0.265</td>\n",
       "      <td>0.188591</td>\n",
       "      <td>0.366782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>10</th>\n",
       "      <td>CKD(t=0)</td>\n",
       "      <td>3.5</td>\n",
       "      <td>3.5</td>\n",
       "      <td>100</td>\n",
       "      <td>16</td>\n",
       "      <td>0.160000</td>\n",
       "      <td>84</td>\n",
       "      <td>0.840000</td>\n",
       "      <td>0.16</td>\n",
       "      <td>0.420</td>\n",
       "      <td>-0.965081</td>\n",
       "      <td>0.366782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>11</th>\n",
       "      <td>CKD(t=0)</td>\n",
       "      <td>4.0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>27</td>\n",
       "      <td>13</td>\n",
       "      <td>0.481481</td>\n",
       "      <td>14</td>\n",
       "      <td>0.518519</td>\n",
       "      <td>0.13</td>\n",
       "      <td>0.070</td>\n",
       "      <td>0.619039</td>\n",
       "      <td>0.366782</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>12</th>\n",
       "      <td>Dieabetes(t=0)</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>196</td>\n",
       "      <td>70</td>\n",
       "      <td>0.357143</td>\n",
       "      <td>126</td>\n",
       "      <td>0.642857</td>\n",
       "      <td>0.70</td>\n",
       "      <td>0.630</td>\n",
       "      <td>0.105361</td>\n",
       "      <td>0.022056</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>13</th>\n",
       "      <td>Dieabetes(t=0)</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>104</td>\n",
       "      <td>30</td>\n",
       "      <td>0.288462</td>\n",
       "      <td>74</td>\n",
       "      <td>0.711538</td>\n",
       "      <td>0.30</td>\n",
       "      <td>0.370</td>\n",
       "      <td>-0.209721</td>\n",
       "      <td>0.022056</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>14</th>\n",
       "      <td>Cholestrol(t=0)</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>201</td>\n",
       "      <td>71</td>\n",
       "      <td>0.353234</td>\n",
       "      <td>130</td>\n",
       "      <td>0.646766</td>\n",
       "      <td>0.71</td>\n",
       "      <td>0.650</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>0.016581</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>15</th>\n",
       "      <td>Cholestrol(t=0)</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>99</td>\n",
       "      <td>29</td>\n",
       "      <td>0.292929</td>\n",
       "      <td>70</td>\n",
       "      <td>0.707071</td>\n",
       "      <td>0.29</td>\n",
       "      <td>0.350</td>\n",
       "      <td>-0.188052</td>\n",
       "      <td>0.016581</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>16</th>\n",
       "      <td>Hyper Tension(t=0)</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>191</td>\n",
       "      <td>68</td>\n",
       "      <td>0.356021</td>\n",
       "      <td>123</td>\n",
       "      <td>0.643979</td>\n",
       "      <td>0.68</td>\n",
       "      <td>0.615</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>0.018551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>17</th>\n",
       "      <td>Hyper Tension(t=0)</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>109</td>\n",
       "      <td>32</td>\n",
       "      <td>0.293578</td>\n",
       "      <td>77</td>\n",
       "      <td>0.706422</td>\n",
       "      <td>0.32</td>\n",
       "      <td>0.385</td>\n",
       "      <td>-0.184922</td>\n",
       "      <td>0.018551</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>18</th>\n",
       "      <td>Hemoglobin(t=0)</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>235</td>\n",
       "      <td>69</td>\n",
       "      <td>0.293617</td>\n",
       "      <td>166</td>\n",
       "      <td>0.706383</td>\n",
       "      <td>0.69</td>\n",
       "      <td>0.830</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>0.109971</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>19</th>\n",
       "      <td>Hemoglobin(t=0)</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>65</td>\n",
       "      <td>31</td>\n",
       "      <td>0.476923</td>\n",
       "      <td>34</td>\n",
       "      <td>0.523077</td>\n",
       "      <td>0.31</td>\n",
       "      <td>0.170</td>\n",
       "      <td>0.600774</td>\n",
       "      <td>0.109971</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>20</th>\n",
       "      <td>Glucose_lastObs</td>\n",
       "      <td>3.5</td>\n",
       "      <td>5.6</td>\n",
       "      <td>80</td>\n",
       "      <td>20</td>\n",
       "      <td>0.250000</td>\n",
       "      <td>60</td>\n",
       "      <td>0.750000</td>\n",
       "      <td>0.20</td>\n",
       "      <td>0.300</td>\n",
       "      <td>-0.405465</td>\n",
       "      <td>0.169468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>21</th>\n",
       "      <td>Glucose_lastObs</td>\n",
       "      <td>5.7</td>\n",
       "      <td>6.3</td>\n",
       "      <td>72</td>\n",
       "      <td>21</td>\n",
       "      <td>0.291667</td>\n",
       "      <td>51</td>\n",
       "      <td>0.708333</td>\n",
       "      <td>0.21</td>\n",
       "      <td>0.255</td>\n",
       "      <td>-0.194156</td>\n",
       "      <td>0.169468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>22</th>\n",
       "      <td>Glucose_lastObs</td>\n",
       "      <td>6.4</td>\n",
       "      <td>7.4</td>\n",
       "      <td>75</td>\n",
       "      <td>23</td>\n",
       "      <td>0.306667</td>\n",
       "      <td>52</td>\n",
       "      <td>0.693333</td>\n",
       "      <td>0.23</td>\n",
       "      <td>0.260</td>\n",
       "      <td>-0.122602</td>\n",
       "      <td>0.169468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>Glucose_lastObs</td>\n",
       "      <td>7.5</td>\n",
       "      <td>16.4</td>\n",
       "      <td>73</td>\n",
       "      <td>36</td>\n",
       "      <td>0.493151</td>\n",
       "      <td>37</td>\n",
       "      <td>0.506849</td>\n",
       "      <td>0.36</td>\n",
       "      <td>0.185</td>\n",
       "      <td>0.665748</td>\n",
       "      <td>0.169468</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>24</th>\n",
       "      <td>Lipoprotein_lastObs</td>\n",
       "      <td>26.6</td>\n",
       "      <td>59.2</td>\n",
       "      <td>51</td>\n",
       "      <td>11</td>\n",
       "      <td>0.215686</td>\n",
       "      <td>40</td>\n",
       "      <td>0.784314</td>\n",
       "      <td>0.11</td>\n",
       "      <td>0.200</td>\n",
       "      <td>-0.597837</td>\n",
       "      <td>0.101544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>25</th>\n",
       "      <td>Lipoprotein_lastObs</td>\n",
       "      <td>59.3</td>\n",
       "      <td>72.2</td>\n",
       "      <td>49</td>\n",
       "      <td>15</td>\n",
       "      <td>0.306122</td>\n",
       "      <td>34</td>\n",
       "      <td>0.693878</td>\n",
       "      <td>0.15</td>\n",
       "      <td>0.170</td>\n",
       "      <td>-0.125163</td>\n",
       "      <td>0.101544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>Lipoprotein_lastObs</td>\n",
       "      <td>72.3</td>\n",
       "      <td>83.3</td>\n",
       "      <td>51</td>\n",
       "      <td>16</td>\n",
       "      <td>0.313725</td>\n",
       "      <td>35</td>\n",
       "      <td>0.686275</td>\n",
       "      <td>0.16</td>\n",
       "      <td>0.175</td>\n",
       "      <td>-0.089612</td>\n",
       "      <td>0.101544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>27</th>\n",
       "      <td>Lipoprotein_lastObs</td>\n",
       "      <td>83.5</td>\n",
       "      <td>93.3</td>\n",
       "      <td>49</td>\n",
       "      <td>17</td>\n",
       "      <td>0.346939</td>\n",
       "      <td>32</td>\n",
       "      <td>0.653061</td>\n",
       "      <td>0.17</td>\n",
       "      <td>0.160</td>\n",
       "      <td>0.060625</td>\n",
       "      <td>0.101544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>28</th>\n",
       "      <td>Lipoprotein_lastObs</td>\n",
       "      <td>93.5</td>\n",
       "      <td>112.5</td>\n",
       "      <td>50</td>\n",
       "      <td>19</td>\n",
       "      <td>0.380000</td>\n",
       "      <td>31</td>\n",
       "      <td>0.620000</td>\n",
       "      <td>0.19</td>\n",
       "      <td>0.155</td>\n",
       "      <td>0.203599</td>\n",
       "      <td>0.101544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>Lipoprotein_lastObs</td>\n",
       "      <td>112.7</td>\n",
       "      <td>198.5</td>\n",
       "      <td>50</td>\n",
       "      <td>22</td>\n",
       "      <td>0.440000</td>\n",
       "      <td>28</td>\n",
       "      <td>0.560000</td>\n",
       "      <td>0.22</td>\n",
       "      <td>0.140</td>\n",
       "      <td>0.451985</td>\n",
       "      <td>0.101544</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>30</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>92.0</td>\n",
       "      <td>117.6</td>\n",
       "      <td>43</td>\n",
       "      <td>10</td>\n",
       "      <td>0.232558</td>\n",
       "      <td>33</td>\n",
       "      <td>0.767442</td>\n",
       "      <td>0.10</td>\n",
       "      <td>0.165</td>\n",
       "      <td>-0.500775</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>31</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>117.8</td>\n",
       "      <td>124.6</td>\n",
       "      <td>44</td>\n",
       "      <td>11</td>\n",
       "      <td>0.250000</td>\n",
       "      <td>33</td>\n",
       "      <td>0.750000</td>\n",
       "      <td>0.11</td>\n",
       "      <td>0.165</td>\n",
       "      <td>-0.405465</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>32</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>124.7</td>\n",
       "      <td>129.7</td>\n",
       "      <td>42</td>\n",
       "      <td>11</td>\n",
       "      <td>0.261905</td>\n",
       "      <td>31</td>\n",
       "      <td>0.738095</td>\n",
       "      <td>0.11</td>\n",
       "      <td>0.155</td>\n",
       "      <td>-0.342945</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>33</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>130.1</td>\n",
       "      <td>134.8</td>\n",
       "      <td>43</td>\n",
       "      <td>14</td>\n",
       "      <td>0.325581</td>\n",
       "      <td>29</td>\n",
       "      <td>0.674419</td>\n",
       "      <td>0.14</td>\n",
       "      <td>0.145</td>\n",
       "      <td>-0.035091</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>34</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>135.0</td>\n",
       "      <td>140.9</td>\n",
       "      <td>42</td>\n",
       "      <td>15</td>\n",
       "      <td>0.357143</td>\n",
       "      <td>27</td>\n",
       "      <td>0.642857</td>\n",
       "      <td>0.15</td>\n",
       "      <td>0.135</td>\n",
       "      <td>0.105361</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>35</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>141.0</td>\n",
       "      <td>148.9</td>\n",
       "      <td>43</td>\n",
       "      <td>16</td>\n",
       "      <td>0.372093</td>\n",
       "      <td>27</td>\n",
       "      <td>0.627907</td>\n",
       "      <td>0.16</td>\n",
       "      <td>0.135</td>\n",
       "      <td>0.169899</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>36</th>\n",
       "      <td>SBP_lastObs</td>\n",
       "      <td>149.0</td>\n",
       "      <td>176.3</td>\n",
       "      <td>43</td>\n",
       "      <td>23</td>\n",
       "      <td>0.534884</td>\n",
       "      <td>20</td>\n",
       "      <td>0.465116</td>\n",
       "      <td>0.23</td>\n",
       "      <td>0.100</td>\n",
       "      <td>0.832909</td>\n",
       "      <td>0.184565</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>HGB_lastObs</td>\n",
       "      <td>9.3</td>\n",
       "      <td>12.6</td>\n",
       "      <td>77</td>\n",
       "      <td>36</td>\n",
       "      <td>0.467532</td>\n",
       "      <td>41</td>\n",
       "      <td>0.532468</td>\n",
       "      <td>0.36</td>\n",
       "      <td>0.205</td>\n",
       "      <td>0.563094</td>\n",
       "      <td>0.164081</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>38</th>\n",
       "      <td>HGB_lastObs</td>\n",
       "      <td>12.7</td>\n",
       "      <td>13.9</td>\n",
       "      <td>76</td>\n",
       "      <td>27</td>\n",
       "      <td>0.355263</td>\n",
       "      <td>49</td>\n",
       "      <td>0.644737</td>\n",
       "      <td>0.27</td>\n",
       "      <td>0.245</td>\n",
       "      <td>0.097164</td>\n",
       "      <td>0.164081</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>39</th>\n",
       "      <td>HGB_lastObs</td>\n",
       "      <td>14.0</td>\n",
       "      <td>15.2</td>\n",
       "      <td>79</td>\n",
       "      <td>21</td>\n",
       "      <td>0.265823</td>\n",
       "      <td>58</td>\n",
       "      <td>0.734177</td>\n",
       "      <td>0.21</td>\n",
       "      <td>0.290</td>\n",
       "      <td>-0.322773</td>\n",
       "      <td>0.164081</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>40</th>\n",
       "      <td>HGB_lastObs</td>\n",
       "      <td>15.3</td>\n",
       "      <td>18.3</td>\n",
       "      <td>68</td>\n",
       "      <td>16</td>\n",
       "      <td>0.235294</td>\n",
       "      <td>52</td>\n",
       "      <td>0.764706</td>\n",
       "      <td>0.16</td>\n",
       "      <td>0.260</td>\n",
       "      <td>-0.485508</td>\n",
       "      <td>0.164081</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>41</th>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>184</td>\n",
       "      <td>55</td>\n",
       "      <td>0.298913</td>\n",
       "      <td>129</td>\n",
       "      <td>0.701087</td>\n",
       "      <td>0.55</td>\n",
       "      <td>0.645</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.037664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>42</th>\n",
       "      <td>Medication Period-Diabetes Type 2</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>116</td>\n",
       "      <td>45</td>\n",
       "      <td>0.387931</td>\n",
       "      <td>71</td>\n",
       "      <td>0.612069</td>\n",
       "      <td>0.45</td>\n",
       "      <td>0.355</td>\n",
       "      <td>0.237130</td>\n",
       "      <td>0.037664</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>43</th>\n",
       "      <td>Medication Period-High Blood Cholestrol</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>71</td>\n",
       "      <td>23</td>\n",
       "      <td>0.323944</td>\n",
       "      <td>48</td>\n",
       "      <td>0.676056</td>\n",
       "      <td>0.23</td>\n",
       "      <td>0.240</td>\n",
       "      <td>-0.042560</td>\n",
       "      <td>0.000556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>44</th>\n",
       "      <td>Medication Period-High Blood Cholestrol</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>229</td>\n",
       "      <td>77</td>\n",
       "      <td>0.336245</td>\n",
       "      <td>152</td>\n",
       "      <td>0.663755</td>\n",
       "      <td>0.77</td>\n",
       "      <td>0.760</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>0.000556</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>45</th>\n",
       "      <td>Medication Period-Hyper Tension</td>\n",
       "      <td>0.0</td>\n",
       "      <td>0.0</td>\n",
       "      <td>128</td>\n",
       "      <td>37</td>\n",
       "      <td>0.289062</td>\n",
       "      <td>91</td>\n",
       "      <td>0.710938</td>\n",
       "      <td>0.37</td>\n",
       "      <td>0.455</td>\n",
       "      <td>-0.206794</td>\n",
       "      <td>0.029897</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>46</th>\n",
       "      <td>Medication Period-Hyper Tension</td>\n",
       "      <td>1.0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>172</td>\n",
       "      <td>63</td>\n",
       "      <td>0.366279</td>\n",
       "      <td>109</td>\n",
       "      <td>0.633721</td>\n",
       "      <td>0.63</td>\n",
       "      <td>0.545</td>\n",
       "      <td>0.144934</td>\n",
       "      <td>0.029897</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                                   VAR_NAME     MIN_VALUE     MAX_VALUE  \\\n",
       "0                                    gender        Female        Female   \n",
       "1                                    gender          Male          Male   \n",
       "2                                 Age Group         40-50         40-50   \n",
       "3                                 Age Group         51-60         51-60   \n",
       "4                                 Age Group         61-70         61-70   \n",
       "5                                 Age Group         71-80         71-80   \n",
       "6                                 Age Group  81 and Above  81 and Above   \n",
       "7                                  CKD(t=0)           1.0           1.0   \n",
       "8                                  CKD(t=0)           2.0           2.0   \n",
       "9                                  CKD(t=0)           3.0           3.0   \n",
       "10                                 CKD(t=0)           3.5           3.5   \n",
       "11                                 CKD(t=0)           4.0           4.0   \n",
       "12                           Dieabetes(t=0)             0             0   \n",
       "13                           Dieabetes(t=0)             1             1   \n",
       "14                          Cholestrol(t=0)             0             0   \n",
       "15                          Cholestrol(t=0)             1             1   \n",
       "16                       Hyper Tension(t=0)             0             0   \n",
       "17                       Hyper Tension(t=0)             1             1   \n",
       "18                          Hemoglobin(t=0)             0             0   \n",
       "19                          Hemoglobin(t=0)             1             1   \n",
       "20                          Glucose_lastObs           3.5           5.6   \n",
       "21                          Glucose_lastObs           5.7           6.3   \n",
       "22                          Glucose_lastObs           6.4           7.4   \n",
       "23                          Glucose_lastObs           7.5          16.4   \n",
       "24                      Lipoprotein_lastObs          26.6          59.2   \n",
       "25                      Lipoprotein_lastObs          59.3          72.2   \n",
       "26                      Lipoprotein_lastObs          72.3          83.3   \n",
       "27                      Lipoprotein_lastObs          83.5          93.3   \n",
       "28                      Lipoprotein_lastObs          93.5         112.5   \n",
       "29                      Lipoprotein_lastObs         112.7         198.5   \n",
       "30                              SBP_lastObs          92.0         117.6   \n",
       "31                              SBP_lastObs         117.8         124.6   \n",
       "32                              SBP_lastObs         124.7         129.7   \n",
       "33                              SBP_lastObs         130.1         134.8   \n",
       "34                              SBP_lastObs         135.0         140.9   \n",
       "35                              SBP_lastObs         141.0         148.9   \n",
       "36                              SBP_lastObs         149.0         176.3   \n",
       "37                              HGB_lastObs           9.3          12.6   \n",
       "38                              HGB_lastObs          12.7          13.9   \n",
       "39                              HGB_lastObs          14.0          15.2   \n",
       "40                              HGB_lastObs          15.3          18.3   \n",
       "41        Medication Period-Diabetes Type 2           0.0           0.0   \n",
       "42        Medication Period-Diabetes Type 2           1.0           1.0   \n",
       "43  Medication Period-High Blood Cholestrol           0.0           0.0   \n",
       "44  Medication Period-High Blood Cholestrol           1.0           1.0   \n",
       "45          Medication Period-Hyper Tension           0.0           0.0   \n",
       "46          Medication Period-Hyper Tension           1.0           1.0   \n",
       "\n",
       "    COUNT  EVENT  EVENT_RATE  NONEVENT  NON_EVENT_RATE  DIST_EVENT  \\\n",
       "0     176     44    0.250000       132        0.750000        0.44   \n",
       "1     124     56    0.451613        68        0.548387        0.56   \n",
       "2      10      4    0.400000         6        0.600000        0.04   \n",
       "3      37     14    0.378378        23        0.621622        0.14   \n",
       "4      93     35    0.376344        58        0.623656        0.35   \n",
       "5     116     35    0.301724        81        0.698276        0.35   \n",
       "6      44     12    0.272727        32        0.727273        0.12   \n",
       "7      17      8    0.470588         9        0.529412        0.08   \n",
       "8      71     31    0.436620        40        0.563380        0.31   \n",
       "9      85     32    0.376471        53        0.623529        0.32   \n",
       "10    100     16    0.160000        84        0.840000        0.16   \n",
       "11     27     13    0.481481        14        0.518519        0.13   \n",
       "12    196     70    0.357143       126        0.642857        0.70   \n",
       "13    104     30    0.288462        74        0.711538        0.30   \n",
       "14    201     71    0.353234       130        0.646766        0.71   \n",
       "15     99     29    0.292929        70        0.707071        0.29   \n",
       "16    191     68    0.356021       123        0.643979        0.68   \n",
       "17    109     32    0.293578        77        0.706422        0.32   \n",
       "18    235     69    0.293617       166        0.706383        0.69   \n",
       "19     65     31    0.476923        34        0.523077        0.31   \n",
       "20     80     20    0.250000        60        0.750000        0.20   \n",
       "21     72     21    0.291667        51        0.708333        0.21   \n",
       "22     75     23    0.306667        52        0.693333        0.23   \n",
       "23     73     36    0.493151        37        0.506849        0.36   \n",
       "24     51     11    0.215686        40        0.784314        0.11   \n",
       "25     49     15    0.306122        34        0.693878        0.15   \n",
       "26     51     16    0.313725        35        0.686275        0.16   \n",
       "27     49     17    0.346939        32        0.653061        0.17   \n",
       "28     50     19    0.380000        31        0.620000        0.19   \n",
       "29     50     22    0.440000        28        0.560000        0.22   \n",
       "30     43     10    0.232558        33        0.767442        0.10   \n",
       "31     44     11    0.250000        33        0.750000        0.11   \n",
       "32     42     11    0.261905        31        0.738095        0.11   \n",
       "33     43     14    0.325581        29        0.674419        0.14   \n",
       "34     42     15    0.357143        27        0.642857        0.15   \n",
       "35     43     16    0.372093        27        0.627907        0.16   \n",
       "36     43     23    0.534884        20        0.465116        0.23   \n",
       "37     77     36    0.467532        41        0.532468        0.36   \n",
       "38     76     27    0.355263        49        0.644737        0.27   \n",
       "39     79     21    0.265823        58        0.734177        0.21   \n",
       "40     68     16    0.235294        52        0.764706        0.16   \n",
       "41    184     55    0.298913       129        0.701087        0.55   \n",
       "42    116     45    0.387931        71        0.612069        0.45   \n",
       "43     71     23    0.323944        48        0.676056        0.23   \n",
       "44    229     77    0.336245       152        0.663755        0.77   \n",
       "45    128     37    0.289062        91        0.710938        0.37   \n",
       "46    172     63    0.366279       109        0.633721        0.63   \n",
       "\n",
       "    DIST_NON_EVENT       WOE        IV  \n",
       "0            0.660 -0.405465  0.198980  \n",
       "1            0.340  0.498991  0.198980  \n",
       "2            0.030  0.287682  0.038612  \n",
       "3            0.115  0.196710  0.038612  \n",
       "4            0.290  0.188052  0.038612  \n",
       "5            0.405 -0.145954  0.038612  \n",
       "6            0.160 -0.287682  0.038612  \n",
       "7            0.045  0.575364  0.366782  \n",
       "8            0.200  0.438255  0.366782  \n",
       "9            0.265  0.188591  0.366782  \n",
       "10           0.420 -0.965081  0.366782  \n",
       "11           0.070  0.619039  0.366782  \n",
       "12           0.630  0.105361  0.022056  \n",
       "13           0.370 -0.209721  0.022056  \n",
       "14           0.650  0.088293  0.016581  \n",
       "15           0.350 -0.188052  0.016581  \n",
       "16           0.615  0.100471  0.018551  \n",
       "17           0.385 -0.184922  0.018551  \n",
       "18           0.830 -0.184734  0.109971  \n",
       "19           0.170  0.600774  0.109971  \n",
       "20           0.300 -0.405465  0.169468  \n",
       "21           0.255 -0.194156  0.169468  \n",
       "22           0.260 -0.122602  0.169468  \n",
       "23           0.185  0.665748  0.169468  \n",
       "24           0.200 -0.597837  0.101544  \n",
       "25           0.170 -0.125163  0.101544  \n",
       "26           0.175 -0.089612  0.101544  \n",
       "27           0.160  0.060625  0.101544  \n",
       "28           0.155  0.203599  0.101544  \n",
       "29           0.140  0.451985  0.101544  \n",
       "30           0.165 -0.500775  0.184565  \n",
       "31           0.165 -0.405465  0.184565  \n",
       "32           0.155 -0.342945  0.184565  \n",
       "33           0.145 -0.035091  0.184565  \n",
       "34           0.135  0.105361  0.184565  \n",
       "35           0.135  0.169899  0.184565  \n",
       "36           0.100  0.832909  0.184565  \n",
       "37           0.205  0.563094  0.164081  \n",
       "38           0.245  0.097164  0.164081  \n",
       "39           0.290 -0.322773  0.164081  \n",
       "40           0.260 -0.485508  0.164081  \n",
       "41           0.645 -0.159332  0.037664  \n",
       "42           0.355  0.237130  0.037664  \n",
       "43           0.240 -0.042560  0.000556  \n",
       "44           0.760  0.013072  0.000556  \n",
       "45           0.455 -0.206794  0.029897  \n",
       "46           0.545  0.144934  0.029897  "
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "final_iv.to_csv('output/InformationValue_2.csv')\n",
    "final_iv"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "d2b90383",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "['gender', 'Age Group', 'CKD(t=0)', 'Dieabetes(t=0)', 'Cholestrol(t=0)', 'Hyper Tension(t=0)', 'Hemoglobin(t=0)', 'Glucose_lastObs', 'Lipoprotein_lastObs', 'SBP_lastObs', 'HGB_lastObs', 'Medication Period-Diabetes Type 2', 'Medication Period-High Blood Cholestrol', 'Medication Period-Hyper Tension']\n"
     ]
    }
   ],
   "source": [
    "transform_prefix = 'new_'\n",
    "transform_vars_list=new_reduced_feature\n",
    "print(transform_vars_list)\n",
    "\n",
    "for var in transform_vars_list:\n",
    "    small_df = final_iv[final_iv['VAR_NAME'] == var]\n",
    "    transform_dict = dict(zip(small_df.MAX_VALUE,small_df.WOE))\n",
    "    replace_cmd = ''\n",
    "    replace_cmd1 = ''\n",
    "    for i in sorted(transform_dict.items()):\n",
    "        replace_cmd = replace_cmd + str(i[1]) + str(' if x <= ') + str(i[0]) + ' else '\n",
    "        replace_cmd1 = replace_cmd1 + str(i[1]) + str(' if x == \"') + str(i[0]) + '\" else '\n",
    "    replace_cmd = replace_cmd + '0'\n",
    "    replace_cmd1 = replace_cmd1 + '0'\n",
    "    if replace_cmd != '0':\n",
    "        try:\n",
    "            dframe_outlierremoved_balanced[transform_prefix + var] = dframe_outlierremoved_balanced[var].apply(lambda x: eval(replace_cmd))\n",
    "        except:\n",
    "            dframe_outlierremoved_balanced[transform_prefix + var] = dframe_outlierremoved_balanced[var].apply(lambda x: eval(replace_cmd1))\n",
    "            \n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "2f241a24",
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
       "      <th>race</th>\n",
       "      <th>gender</th>\n",
       "      <th>Age Group</th>\n",
       "      <th>target</th>\n",
       "      <th>CKD(t=0)</th>\n",
       "      <th>Dieabetes(t=0)</th>\n",
       "      <th>Cholestrol(t=0)</th>\n",
       "      <th>Hyper Tension(t=0)</th>\n",
       "      <th>Hemoglobin(t=0)</th>\n",
       "      <th>Creatinine_baseline</th>\n",
       "      <th>...</th>\n",
       "      <th>new_Cholestrol(t=0)</th>\n",
       "      <th>new_Hyper Tension(t=0)</th>\n",
       "      <th>new_Hemoglobin(t=0)</th>\n",
       "      <th>new_Glucose_lastObs</th>\n",
       "      <th>new_Lipoprotein_lastObs</th>\n",
       "      <th>new_SBP_lastObs</th>\n",
       "      <th>new_HGB_lastObs</th>\n",
       "      <th>new_Medication Period-Diabetes Type 2</th>\n",
       "      <th>new_Medication Period-High Blood Cholestrol</th>\n",
       "      <th>new_Medication Period-Hyper Tension</th>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>id</th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "      <th></th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Unknown</td>\n",
       "      <td>Male</td>\n",
       "      <td>61-70</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1.3</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.188052</td>\n",
       "      <td>-0.184922</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>-0.194156</td>\n",
       "      <td>0.451985</td>\n",
       "      <td>0.169899</td>\n",
       "      <td>0.097164</td>\n",
       "      <td>0.237130</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>0.144934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>71-80</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.7</td>\n",
       "      <td>...</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>0.665748</td>\n",
       "      <td>-0.089612</td>\n",
       "      <td>0.832909</td>\n",
       "      <td>0.097164</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>-0.206794</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>51-60</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>2.5</td>\n",
       "      <td>...</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>-0.184922</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>-0.122602</td>\n",
       "      <td>0.060625</td>\n",
       "      <td>0.105361</td>\n",
       "      <td>-0.322773</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>-0.206794</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>White</td>\n",
       "      <td>Male</td>\n",
       "      <td>71-80</td>\n",
       "      <td>0</td>\n",
       "      <td>1.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0.7</td>\n",
       "      <td>...</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>-0.184922</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>-0.194156</td>\n",
       "      <td>0.203599</td>\n",
       "      <td>0.169899</td>\n",
       "      <td>0.097164</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>-0.206794</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>40-50</td>\n",
       "      <td>1</td>\n",
       "      <td>3.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.3</td>\n",
       "      <td>...</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>0.665748</td>\n",
       "      <td>-0.125163</td>\n",
       "      <td>0.832909</td>\n",
       "      <td>0.563094</td>\n",
       "      <td>0.237130</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>0.144934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>Black</td>\n",
       "      <td>Male</td>\n",
       "      <td>71-80</td>\n",
       "      <td>0</td>\n",
       "      <td>3.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>1.5</td>\n",
       "      <td>...</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>0.600774</td>\n",
       "      <td>-0.122602</td>\n",
       "      <td>-0.125163</td>\n",
       "      <td>-0.500775</td>\n",
       "      <td>0.563094</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>0.144934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>6</th>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>51-60</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.8</td>\n",
       "      <td>...</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>-0.122602</td>\n",
       "      <td>-0.597837</td>\n",
       "      <td>0.832909</td>\n",
       "      <td>-0.485508</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>0.144934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>7</th>\n",
       "      <td>Asian</td>\n",
       "      <td>Female</td>\n",
       "      <td>61-70</td>\n",
       "      <td>0</td>\n",
       "      <td>3.5</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.4</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.188052</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>-0.194156</td>\n",
       "      <td>0.451985</td>\n",
       "      <td>-0.500775</td>\n",
       "      <td>-0.485508</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>0.144934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>8</th>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>61-70</td>\n",
       "      <td>1</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0</td>\n",
       "      <td>1</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>2.0</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.188052</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>-0.405465</td>\n",
       "      <td>0.451985</td>\n",
       "      <td>0.169899</td>\n",
       "      <td>-0.485508</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>0.013072</td>\n",
       "      <td>0.144934</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>White</td>\n",
       "      <td>Female</td>\n",
       "      <td>71-80</td>\n",
       "      <td>0</td>\n",
       "      <td>4.0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>0</td>\n",
       "      <td>1.8</td>\n",
       "      <td>...</td>\n",
       "      <td>0.088293</td>\n",
       "      <td>0.100471</td>\n",
       "      <td>-0.184734</td>\n",
       "      <td>-0.194156</td>\n",
       "      <td>-0.125163</td>\n",
       "      <td>-0.405465</td>\n",
       "      <td>0.097164</td>\n",
       "      <td>-0.159332</td>\n",
       "      <td>-0.042560</td>\n",
       "      <td>0.144934</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>10 rows × 36 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "       race  gender Age Group  target CKD(t=0) Dieabetes(t=0) Cholestrol(t=0)  \\\n",
       "id                                                                              \n",
       "0   Unknown    Male     61-70       1      3.0              0               1   \n",
       "1     White  Female     71-80       0      4.0              1               0   \n",
       "2     White  Female     51-60       1      4.0              1               0   \n",
       "3     White    Male     71-80       0      1.0              0               0   \n",
       "4     White  Female     40-50       1      3.0              1               0   \n",
       "5     Black    Male     71-80       0      3.0              0               0   \n",
       "6     White  Female     51-60       0      4.0              1               0   \n",
       "7     Asian  Female     61-70       0      3.5              0               1   \n",
       "8     White  Female     61-70       1      4.0              0               1   \n",
       "9     White  Female     71-80       0      4.0              0               0   \n",
       "\n",
       "   Hyper Tension(t=0) Hemoglobin(t=0)  Creatinine_baseline  ...  \\\n",
       "id                                                          ...   \n",
       "0                   1               0                  1.3  ...   \n",
       "1                   0               0                  1.7  ...   \n",
       "2                   1               0                  2.5  ...   \n",
       "3                   1               0                  0.7  ...   \n",
       "4                   0               0                  1.3  ...   \n",
       "5                   0               1                  1.5  ...   \n",
       "6                   0               0                  1.8  ...   \n",
       "7                   0               0                  1.4  ...   \n",
       "8                   0               0                  2.0  ...   \n",
       "9                   0               0                  1.8  ...   \n",
       "\n",
       "    new_Cholestrol(t=0)  new_Hyper Tension(t=0)  new_Hemoglobin(t=0)  \\\n",
       "id                                                                     \n",
       "0             -0.188052               -0.184922            -0.184734   \n",
       "1              0.088293                0.100471            -0.184734   \n",
       "2              0.088293               -0.184922            -0.184734   \n",
       "3              0.088293               -0.184922            -0.184734   \n",
       "4              0.088293                0.100471            -0.184734   \n",
       "5              0.088293                0.100471             0.600774   \n",
       "6              0.088293                0.100471            -0.184734   \n",
       "7             -0.188052                0.100471            -0.184734   \n",
       "8             -0.188052                0.100471            -0.184734   \n",
       "9              0.088293                0.100471            -0.184734   \n",
       "\n",
       "    new_Glucose_lastObs  new_Lipoprotein_lastObs  new_SBP_lastObs  \\\n",
       "id                                                                  \n",
       "0             -0.194156                 0.451985         0.169899   \n",
       "1              0.665748                -0.089612         0.832909   \n",
       "2             -0.122602                 0.060625         0.105361   \n",
       "3             -0.194156                 0.203599         0.169899   \n",
       "4              0.665748                -0.125163         0.832909   \n",
       "5             -0.122602                -0.125163        -0.500775   \n",
       "6             -0.122602                -0.597837         0.832909   \n",
       "7             -0.194156                 0.451985        -0.500775   \n",
       "8             -0.405465                 0.451985         0.169899   \n",
       "9             -0.194156                -0.125163        -0.405465   \n",
       "\n",
       "    new_HGB_lastObs  new_Medication Period-Diabetes Type 2  \\\n",
       "id                                                           \n",
       "0          0.097164                               0.237130   \n",
       "1          0.097164                              -0.159332   \n",
       "2         -0.322773                              -0.159332   \n",
       "3          0.097164                              -0.159332   \n",
       "4          0.563094                               0.237130   \n",
       "5          0.563094                              -0.159332   \n",
       "6         -0.485508                              -0.159332   \n",
       "7         -0.485508                              -0.159332   \n",
       "8         -0.485508                              -0.159332   \n",
       "9          0.097164                              -0.159332   \n",
       "\n",
       "    new_Medication Period-High Blood Cholestrol  \\\n",
       "id                                                \n",
       "0                                      0.013072   \n",
       "1                                      0.013072   \n",
       "2                                      0.013072   \n",
       "3                                      0.013072   \n",
       "4                                      0.013072   \n",
       "5                                      0.013072   \n",
       "6                                      0.013072   \n",
       "7                                      0.013072   \n",
       "8                                      0.013072   \n",
       "9                                     -0.042560   \n",
       "\n",
       "   new_Medication Period-Hyper Tension  \n",
       "id                                      \n",
       "0                             0.144934  \n",
       "1                            -0.206794  \n",
       "2                            -0.206794  \n",
       "3                            -0.206794  \n",
       "4                             0.144934  \n",
       "5                             0.144934  \n",
       "6                             0.144934  \n",
       "7                             0.144934  \n",
       "8                             0.144934  \n",
       "9                             0.144934  \n",
       "\n",
       "[10 rows x 36 columns]"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dframe_outlierremoved_balanced.to_csv('output/Prediction-with-WOE.csv')\n",
    "dframe_outlierremoved_balanced.head(10)\n"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ecab42fd",
   "metadata": {},
   "source": [
    "### 4.Predictive Modelling"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "186c8e8d",
   "metadata": {},
   "source": [
    "Methodology : The approach towards modelling is two phased. First a simple logistic regression model is built with features selected by scanning the IV\\WOE results above.\n",
    "An advanced approach will consider the longitudunal data"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3147bce1",
   "metadata": {},
   "source": [
    "#### <font color='green'> Simple Approach : Logistic Regression \\ Random Forest</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "7d939689",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Optimization terminated successfully.\n",
      "         Current function value: 0.559579\n",
      "         Iterations 6\n",
      "                           Logit Regression Results                           \n",
      "==============================================================================\n",
      "Dep. Variable:                 target   No. Observations:                  200\n",
      "Model:                          Logit   Df Residuals:                      192\n",
      "Method:                           MLE   Df Model:                            7\n",
      "Date:                Mon, 11 Apr 2022   Pseudo R-squ.:                  0.1927\n",
      "Time:                        16:12:21   Log-Likelihood:                -111.92\n",
      "converged:                       True   LL-Null:                       -138.63\n",
      "Covariance Type:            nonrobust   LLR p-value:                 3.053e-09\n",
      "===========================================================================================\n",
      "                              coef    std err          z      P>|z|      [0.025      0.975]\n",
      "-------------------------------------------------------------------------------------------\n",
      "new_gender                  0.7845      0.469      1.672      0.095      -0.135       1.704\n",
      "new_Age Group               0.5404      0.876      0.617      0.537      -1.177       2.258\n",
      "new_CKD(t=0)                0.8139      0.296      2.751      0.006       0.234       1.394\n",
      "new_Hemoglobin(t=0)        -0.8284      0.662     -1.252      0.211      -2.125       0.468\n",
      "new_Glucose_lastObs         1.6597      0.445      3.729      0.000       0.787       2.532\n",
      "new_Lipoprotein_lastObs     1.5170      0.538      2.819      0.005       0.462       2.572\n",
      "new_SBP_lastObs             0.7970      0.394      2.022      0.043       0.024       1.570\n",
      "new_HGB_lastObs             1.0415      0.522      1.997      0.046       0.019       2.064\n",
      "===========================================================================================\n"
     ]
    }
   ],
   "source": [
    "# dframe_outlierremoved_balanced.to_csv('output/dframe_outlierremoved_balanced.csv')\n",
    "\n",
    "import MyClassifierToolSet as my_clf\n",
    "\n",
    "features=[ \n",
    "           'new_gender','new_Age Group',\n",
    "           'new_CKD(t=0)','new_Hemoglobin(t=0)',\n",
    "           \n",
    "            'new_Glucose_lastObs', 'new_Lipoprotein_lastObs', 'new_SBP_lastObs', 'new_HGB_lastObs',\n",
    "              \n",
    "         ]\n",
    "\n",
    "target='target'  #'new_Lipoprotein_delta'\n",
    "# Inspect p values\n",
    "dframe_target=dframe_outlierremoved_balanced[target][dframe_outlierremoved_balanced.index.isin(idx_0) | dframe_outlierremoved_balanced.index.isin(idx_1)]\n",
    "dframe_features=dframe_outlierremoved_balanced[features][dframe_outlierremoved_balanced.index.isin(idx_0) | dframe_outlierremoved_balanced.index.isin(idx_1)]\n",
    "\n",
    "import statsmodels.api as sm\n",
    "logit_model=sm.Logit(dframe_target,dframe_features)\n",
    "result=logit_model.fit()\n",
    "print(result.summary())\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "f849294a",
   "metadata": {
    "scrolled": false
   },
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Training Records  and Test Records [140, 60]\n",
      "\n",
      " minority observation in test data\n",
      "1    32\n",
      "0    28\n",
      "Name: target, dtype: int64\n",
      "\n",
      "\n",
      "\n",
      "Optimization terminated successfully.\n",
      "         Current function value: 0.547620\n",
      "         Iterations 6\n",
      "                           Logit Regression Results                           \n",
      "==============================================================================\n",
      "Dep. Variable:                 target   No. Observations:                  140\n",
      "Model:                          Logit   Df Residuals:                      134\n",
      "Method:                           MLE   Df Model:                            5\n",
      "Date:                Mon, 11 Apr 2022   Pseudo R-squ.:                  0.2095\n",
      "Time:                        16:12:21   Log-Likelihood:                -76.667\n",
      "converged:                       True   LL-Null:                       -96.983\n",
      "Covariance Type:            nonrobust   LLR p-value:                 1.113e-07\n",
      "===========================================================================================\n",
      "                              coef    std err          z      P>|z|      [0.025      0.975]\n",
      "-------------------------------------------------------------------------------------------\n",
      "new_gender                  0.6655      0.515      1.293      0.196      -0.343       1.674\n",
      "new_CKD(t=0)                0.7339      0.351      2.089      0.037       0.045       1.422\n",
      "new_Glucose_lastObs         1.8465      0.586      3.149      0.002       0.697       2.996\n",
      "new_Lipoprotein_lastObs     1.3560      0.684      1.984      0.047       0.016       2.696\n",
      "new_SBP_lastObs             1.3729      0.489      2.806      0.005       0.414       2.332\n",
      "new_HGB_lastObs             0.9199      0.571      1.610      0.107      -0.200       2.040\n",
      "===========================================================================================\n",
      "----------------Test data Performance-----------------------\n",
      "\n",
      "Precision: 0.724138\n",
      "Recall: 0.656250\n",
      "F1 score: 0.688525\n",
      "\n",
      " Logistic Regression Model Called with Train-Test split as 70-30\n",
      "\n",
      " Training Records  and Test Records [210, 90]\n",
      "\n",
      " minority observation in test data\n",
      "0    58\n",
      "1    32\n",
      "Name: target, dtype: int64\n",
      "\n",
      "\n",
      "\n",
      "\n",
      " Accuracy Train ------- Accuracy Test------------ AUC Train-------AUC Test--------\n",
      "0.72 0.7 (0.7673467274233636, 0.7809806034482758)\n",
      "\n",
      "\n"
     ]
    },
    {
     "data": {
      "text/html": [
       "\n",
       "    <div class=\"bk-root\">\n",
       "        <a href=\"https://bokeh.org\" target=\"_blank\" class=\"bk-logo bk-logo-small bk-logo-notebook\"></a>\n",
       "        <span id=\"1002\">Loading BokehJS ...</span>\n",
       "    </div>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "\n",
       "(function(root) {\n",
       "  function now() {\n",
       "    return new Date();\n",
       "  }\n",
       "\n",
       "  var force = true;\n",
       "\n",
       "  if (typeof root._bokeh_onload_callbacks === \"undefined\" || force === true) {\n",
       "    root._bokeh_onload_callbacks = [];\n",
       "    root._bokeh_is_loading = undefined;\n",
       "  }\n",
       "\n",
       "  var JS_MIME_TYPE = 'application/javascript';\n",
       "  var HTML_MIME_TYPE = 'text/html';\n",
       "  var EXEC_MIME_TYPE = 'application/vnd.bokehjs_exec.v0+json';\n",
       "  var CLASS_NAME = 'output_bokeh rendered_html';\n",
       "\n",
       "  /**\n",
       "   * Render data to the DOM node\n",
       "   */\n",
       "  function render(props, node) {\n",
       "    var script = document.createElement(\"script\");\n",
       "    node.appendChild(script);\n",
       "  }\n",
       "\n",
       "  /**\n",
       "   * Handle when an output is cleared or removed\n",
       "   */\n",
       "  function handleClearOutput(event, handle) {\n",
       "    var cell = handle.cell;\n",
       "\n",
       "    var id = cell.output_area._bokeh_element_id;\n",
       "    var server_id = cell.output_area._bokeh_server_id;\n",
       "    // Clean up Bokeh references\n",
       "    if (id != null && id in Bokeh.index) {\n",
       "      Bokeh.index[id].model.document.clear();\n",
       "      delete Bokeh.index[id];\n",
       "    }\n",
       "\n",
       "    if (server_id !== undefined) {\n",
       "      // Clean up Bokeh references\n",
       "      var cmd = \"from bokeh.io.state import curstate; print(curstate().uuid_to_server['\" + server_id + \"'].get_sessions()[0].document.roots[0]._id)\";\n",
       "      cell.notebook.kernel.execute(cmd, {\n",
       "        iopub: {\n",
       "          output: function(msg) {\n",
       "            var id = msg.content.text.trim();\n",
       "            if (id in Bokeh.index) {\n",
       "              Bokeh.index[id].model.document.clear();\n",
       "              delete Bokeh.index[id];\n",
       "            }\n",
       "          }\n",
       "        }\n",
       "      });\n",
       "      // Destroy server and session\n",
       "      var cmd = \"import bokeh.io.notebook as ion; ion.destroy_server('\" + server_id + \"')\";\n",
       "      cell.notebook.kernel.execute(cmd);\n",
       "    }\n",
       "  }\n",
       "\n",
       "  /**\n",
       "   * Handle when a new output is added\n",
       "   */\n",
       "  function handleAddOutput(event, handle) {\n",
       "    var output_area = handle.output_area;\n",
       "    var output = handle.output;\n",
       "\n",
       "    // limit handleAddOutput to display_data with EXEC_MIME_TYPE content only\n",
       "    if ((output.output_type != \"display_data\") || (!Object.prototype.hasOwnProperty.call(output.data, EXEC_MIME_TYPE))) {\n",
       "      return\n",
       "    }\n",
       "\n",
       "    var toinsert = output_area.element.find(\".\" + CLASS_NAME.split(' ')[0]);\n",
       "\n",
       "    if (output.metadata[EXEC_MIME_TYPE][\"id\"] !== undefined) {\n",
       "      toinsert[toinsert.length - 1].firstChild.textContent = output.data[JS_MIME_TYPE];\n",
       "      // store reference to embed id on output_area\n",
       "      output_area._bokeh_element_id = output.metadata[EXEC_MIME_TYPE][\"id\"];\n",
       "    }\n",
       "    if (output.metadata[EXEC_MIME_TYPE][\"server_id\"] !== undefined) {\n",
       "      var bk_div = document.createElement(\"div\");\n",
       "      bk_div.innerHTML = output.data[HTML_MIME_TYPE];\n",
       "      var script_attrs = bk_div.children[0].attributes;\n",
       "      for (var i = 0; i < script_attrs.length; i++) {\n",
       "        toinsert[toinsert.length - 1].firstChild.setAttribute(script_attrs[i].name, script_attrs[i].value);\n",
       "        toinsert[toinsert.length - 1].firstChild.textContent = bk_div.children[0].textContent\n",
       "      }\n",
       "      // store reference to server id on output_area\n",
       "      output_area._bokeh_server_id = output.metadata[EXEC_MIME_TYPE][\"server_id\"];\n",
       "    }\n",
       "  }\n",
       "\n",
       "  function register_renderer(events, OutputArea) {\n",
       "\n",
       "    function append_mime(data, metadata, element) {\n",
       "      // create a DOM node to render to\n",
       "      var toinsert = this.create_output_subarea(\n",
       "        metadata,\n",
       "        CLASS_NAME,\n",
       "        EXEC_MIME_TYPE\n",
       "      );\n",
       "      this.keyboard_manager.register_events(toinsert);\n",
       "      // Render to node\n",
       "      var props = {data: data, metadata: metadata[EXEC_MIME_TYPE]};\n",
       "      render(props, toinsert[toinsert.length - 1]);\n",
       "      element.append(toinsert);\n",
       "      return toinsert\n",
       "    }\n",
       "\n",
       "    /* Handle when an output is cleared or removed */\n",
       "    events.on('clear_output.CodeCell', handleClearOutput);\n",
       "    events.on('delete.Cell', handleClearOutput);\n",
       "\n",
       "    /* Handle when a new output is added */\n",
       "    events.on('output_added.OutputArea', handleAddOutput);\n",
       "\n",
       "    /**\n",
       "     * Register the mime type and append_mime function with output_area\n",
       "     */\n",
       "    OutputArea.prototype.register_mime_type(EXEC_MIME_TYPE, append_mime, {\n",
       "      /* Is output safe? */\n",
       "      safe: true,\n",
       "      /* Index of renderer in `output_area.display_order` */\n",
       "      index: 0\n",
       "    });\n",
       "  }\n",
       "\n",
       "  // register the mime type if in Jupyter Notebook environment and previously unregistered\n",
       "  if (root.Jupyter !== undefined) {\n",
       "    var events = require('base/js/events');\n",
       "    var OutputArea = require('notebook/js/outputarea').OutputArea;\n",
       "\n",
       "    if (OutputArea.prototype.mime_types().indexOf(EXEC_MIME_TYPE) == -1) {\n",
       "      register_renderer(events, OutputArea);\n",
       "    }\n",
       "  }\n",
       "\n",
       "  \n",
       "  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n",
       "    root._bokeh_timeout = Date.now() + 5000;\n",
       "    root._bokeh_failed_load = false;\n",
       "  }\n",
       "\n",
       "  var NB_LOAD_WARNING = {'data': {'text/html':\n",
       "     \"<div style='background-color: #fdd'>\\n\"+\n",
       "     \"<p>\\n\"+\n",
       "     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n",
       "     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n",
       "     \"</p>\\n\"+\n",
       "     \"<ul>\\n\"+\n",
       "     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n",
       "     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n",
       "     \"</ul>\\n\"+\n",
       "     \"<code>\\n\"+\n",
       "     \"from bokeh.resources import INLINE\\n\"+\n",
       "     \"output_notebook(resources=INLINE)\\n\"+\n",
       "     \"</code>\\n\"+\n",
       "     \"</div>\"}};\n",
       "\n",
       "  function display_loaded() {\n",
       "    var el = document.getElementById(\"1002\");\n",
       "    if (el != null) {\n",
       "      el.textContent = \"BokehJS is loading...\";\n",
       "    }\n",
       "    if (root.Bokeh !== undefined) {\n",
       "      if (el != null) {\n",
       "        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n",
       "      }\n",
       "    } else if (Date.now() < root._bokeh_timeout) {\n",
       "      setTimeout(display_loaded, 100)\n",
       "    }\n",
       "  }\n",
       "\n",
       "\n",
       "  function run_callbacks() {\n",
       "    try {\n",
       "      root._bokeh_onload_callbacks.forEach(function(callback) {\n",
       "        if (callback != null)\n",
       "          callback();\n",
       "      });\n",
       "    } finally {\n",
       "      delete root._bokeh_onload_callbacks\n",
       "    }\n",
       "    console.debug(\"Bokeh: all callbacks have finished\");\n",
       "  }\n",
       "\n",
       "  function load_libs(css_urls, js_urls, callback) {\n",
       "    if (css_urls == null) css_urls = [];\n",
       "    if (js_urls == null) js_urls = [];\n",
       "\n",
       "    root._bokeh_onload_callbacks.push(callback);\n",
       "    if (root._bokeh_is_loading > 0) {\n",
       "      console.debug(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n",
       "      return null;\n",
       "    }\n",
       "    if (js_urls == null || js_urls.length === 0) {\n",
       "      run_callbacks();\n",
       "      return null;\n",
       "    }\n",
       "    console.debug(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n",
       "    root._bokeh_is_loading = css_urls.length + js_urls.length;\n",
       "\n",
       "    function on_load() {\n",
       "      root._bokeh_is_loading--;\n",
       "      if (root._bokeh_is_loading === 0) {\n",
       "        console.debug(\"Bokeh: all BokehJS libraries/stylesheets loaded\");\n",
       "        run_callbacks()\n",
       "      }\n",
       "    }\n",
       "\n",
       "    function on_error(url) {\n",
       "      console.error(\"failed to load \" + url);\n",
       "    }\n",
       "\n",
       "    for (let i = 0; i < css_urls.length; i++) {\n",
       "      const url = css_urls[i];\n",
       "      const element = document.createElement(\"link\");\n",
       "      element.onload = on_load;\n",
       "      element.onerror = on_error.bind(null, url);\n",
       "      element.rel = \"stylesheet\";\n",
       "      element.type = \"text/css\";\n",
       "      element.href = url;\n",
       "      console.debug(\"Bokeh: injecting link tag for BokehJS stylesheet: \", url);\n",
       "      document.body.appendChild(element);\n",
       "    }\n",
       "\n",
       "    const hashes = {\"https://cdn.bokeh.org/bokeh/release/bokeh-2.3.2.min.js\": \"XypntL49z55iwGVUW4qsEu83zKL3XEcz0MjuGOQ9SlaaQ68X/g+k1FcioZi7oQAc\", \"https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.3.2.min.js\": \"bEsM86IHGDTLCS0Zod8a8WM6Y4+lafAL/eSiyQcuPzinmWNgNO2/olUF0Z2Dkn5i\", \"https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.3.2.min.js\": \"TX0gSQTdXTTeScqxj6PVQxTiRW8DOoGVwinyi1D3kxv7wuxQ02XkOxv0xwiypcAH\"};\n",
       "\n",
       "    for (let i = 0; i < js_urls.length; i++) {\n",
       "      const url = js_urls[i];\n",
       "      const element = document.createElement('script');\n",
       "      element.onload = on_load;\n",
       "      element.onerror = on_error.bind(null, url);\n",
       "      element.async = false;\n",
       "      element.src = url;\n",
       "      if (url in hashes) {\n",
       "        element.crossOrigin = \"anonymous\";\n",
       "        element.integrity = \"sha384-\" + hashes[url];\n",
       "      }\n",
       "      console.debug(\"Bokeh: injecting script tag for BokehJS library: \", url);\n",
       "      document.head.appendChild(element);\n",
       "    }\n",
       "  };\n",
       "\n",
       "  function inject_raw_css(css) {\n",
       "    const element = document.createElement(\"style\");\n",
       "    element.appendChild(document.createTextNode(css));\n",
       "    document.body.appendChild(element);\n",
       "  }\n",
       "\n",
       "  \n",
       "  var js_urls = [\"https://cdn.bokeh.org/bokeh/release/bokeh-2.3.2.min.js\", \"https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.3.2.min.js\", \"https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.3.2.min.js\"];\n",
       "  var css_urls = [];\n",
       "  \n",
       "\n",
       "  var inline_js = [\n",
       "    function(Bokeh) {\n",
       "      Bokeh.set_log_level(\"info\");\n",
       "    },\n",
       "    function(Bokeh) {\n",
       "    \n",
       "    \n",
       "    }\n",
       "  ];\n",
       "\n",
       "  function run_inline_js() {\n",
       "    \n",
       "    if (root.Bokeh !== undefined || force === true) {\n",
       "      \n",
       "    for (var i = 0; i < inline_js.length; i++) {\n",
       "      inline_js[i].call(root, root.Bokeh);\n",
       "    }\n",
       "    if (force === true) {\n",
       "        display_loaded();\n",
       "      }} else if (Date.now() < root._bokeh_timeout) {\n",
       "      setTimeout(run_inline_js, 100);\n",
       "    } else if (!root._bokeh_failed_load) {\n",
       "      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n",
       "      root._bokeh_failed_load = true;\n",
       "    } else if (force !== true) {\n",
       "      var cell = $(document.getElementById(\"1002\")).parents('.cell').data().cell;\n",
       "      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n",
       "    }\n",
       "\n",
       "  }\n",
       "\n",
       "  if (root._bokeh_is_loading === 0) {\n",
       "    console.debug(\"Bokeh: BokehJS loaded, going straight to plotting\");\n",
       "    run_inline_js();\n",
       "  } else {\n",
       "    load_libs(css_urls, js_urls, function() {\n",
       "      console.debug(\"Bokeh: BokehJS plotting callback run at\", now());\n",
       "      run_inline_js();\n",
       "    });\n",
       "  }\n",
       "}(window));"
      ],
      "application/vnd.bokehjs_load.v0+json": "\n(function(root) {\n  function now() {\n    return new Date();\n  }\n\n  var force = true;\n\n  if (typeof root._bokeh_onload_callbacks === \"undefined\" || force === true) {\n    root._bokeh_onload_callbacks = [];\n    root._bokeh_is_loading = undefined;\n  }\n\n  \n\n  \n  if (typeof (root._bokeh_timeout) === \"undefined\" || force === true) {\n    root._bokeh_timeout = Date.now() + 5000;\n    root._bokeh_failed_load = false;\n  }\n\n  var NB_LOAD_WARNING = {'data': {'text/html':\n     \"<div style='background-color: #fdd'>\\n\"+\n     \"<p>\\n\"+\n     \"BokehJS does not appear to have successfully loaded. If loading BokehJS from CDN, this \\n\"+\n     \"may be due to a slow or bad network connection. Possible fixes:\\n\"+\n     \"</p>\\n\"+\n     \"<ul>\\n\"+\n     \"<li>re-rerun `output_notebook()` to attempt to load from CDN again, or</li>\\n\"+\n     \"<li>use INLINE resources instead, as so:</li>\\n\"+\n     \"</ul>\\n\"+\n     \"<code>\\n\"+\n     \"from bokeh.resources import INLINE\\n\"+\n     \"output_notebook(resources=INLINE)\\n\"+\n     \"</code>\\n\"+\n     \"</div>\"}};\n\n  function display_loaded() {\n    var el = document.getElementById(\"1002\");\n    if (el != null) {\n      el.textContent = \"BokehJS is loading...\";\n    }\n    if (root.Bokeh !== undefined) {\n      if (el != null) {\n        el.textContent = \"BokehJS \" + root.Bokeh.version + \" successfully loaded.\";\n      }\n    } else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(display_loaded, 100)\n    }\n  }\n\n\n  function run_callbacks() {\n    try {\n      root._bokeh_onload_callbacks.forEach(function(callback) {\n        if (callback != null)\n          callback();\n      });\n    } finally {\n      delete root._bokeh_onload_callbacks\n    }\n    console.debug(\"Bokeh: all callbacks have finished\");\n  }\n\n  function load_libs(css_urls, js_urls, callback) {\n    if (css_urls == null) css_urls = [];\n    if (js_urls == null) js_urls = [];\n\n    root._bokeh_onload_callbacks.push(callback);\n    if (root._bokeh_is_loading > 0) {\n      console.debug(\"Bokeh: BokehJS is being loaded, scheduling callback at\", now());\n      return null;\n    }\n    if (js_urls == null || js_urls.length === 0) {\n      run_callbacks();\n      return null;\n    }\n    console.debug(\"Bokeh: BokehJS not loaded, scheduling load and callback at\", now());\n    root._bokeh_is_loading = css_urls.length + js_urls.length;\n\n    function on_load() {\n      root._bokeh_is_loading--;\n      if (root._bokeh_is_loading === 0) {\n        console.debug(\"Bokeh: all BokehJS libraries/stylesheets loaded\");\n        run_callbacks()\n      }\n    }\n\n    function on_error(url) {\n      console.error(\"failed to load \" + url);\n    }\n\n    for (let i = 0; i < css_urls.length; i++) {\n      const url = css_urls[i];\n      const element = document.createElement(\"link\");\n      element.onload = on_load;\n      element.onerror = on_error.bind(null, url);\n      element.rel = \"stylesheet\";\n      element.type = \"text/css\";\n      element.href = url;\n      console.debug(\"Bokeh: injecting link tag for BokehJS stylesheet: \", url);\n      document.body.appendChild(element);\n    }\n\n    const hashes = {\"https://cdn.bokeh.org/bokeh/release/bokeh-2.3.2.min.js\": \"XypntL49z55iwGVUW4qsEu83zKL3XEcz0MjuGOQ9SlaaQ68X/g+k1FcioZi7oQAc\", \"https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.3.2.min.js\": \"bEsM86IHGDTLCS0Zod8a8WM6Y4+lafAL/eSiyQcuPzinmWNgNO2/olUF0Z2Dkn5i\", \"https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.3.2.min.js\": \"TX0gSQTdXTTeScqxj6PVQxTiRW8DOoGVwinyi1D3kxv7wuxQ02XkOxv0xwiypcAH\"};\n\n    for (let i = 0; i < js_urls.length; i++) {\n      const url = js_urls[i];\n      const element = document.createElement('script');\n      element.onload = on_load;\n      element.onerror = on_error.bind(null, url);\n      element.async = false;\n      element.src = url;\n      if (url in hashes) {\n        element.crossOrigin = \"anonymous\";\n        element.integrity = \"sha384-\" + hashes[url];\n      }\n      console.debug(\"Bokeh: injecting script tag for BokehJS library: \", url);\n      document.head.appendChild(element);\n    }\n  };\n\n  function inject_raw_css(css) {\n    const element = document.createElement(\"style\");\n    element.appendChild(document.createTextNode(css));\n    document.body.appendChild(element);\n  }\n\n  \n  var js_urls = [\"https://cdn.bokeh.org/bokeh/release/bokeh-2.3.2.min.js\", \"https://cdn.bokeh.org/bokeh/release/bokeh-widgets-2.3.2.min.js\", \"https://cdn.bokeh.org/bokeh/release/bokeh-tables-2.3.2.min.js\"];\n  var css_urls = [];\n  \n\n  var inline_js = [\n    function(Bokeh) {\n      Bokeh.set_log_level(\"info\");\n    },\n    function(Bokeh) {\n    \n    \n    }\n  ];\n\n  function run_inline_js() {\n    \n    if (root.Bokeh !== undefined || force === true) {\n      \n    for (var i = 0; i < inline_js.length; i++) {\n      inline_js[i].call(root, root.Bokeh);\n    }\n    if (force === true) {\n        display_loaded();\n      }} else if (Date.now() < root._bokeh_timeout) {\n      setTimeout(run_inline_js, 100);\n    } else if (!root._bokeh_failed_load) {\n      console.log(\"Bokeh: BokehJS failed to load within specified timeout.\");\n      root._bokeh_failed_load = true;\n    } else if (force !== true) {\n      var cell = $(document.getElementById(\"1002\")).parents('.cell').data().cell;\n      cell.output_area.append_execute_result(NB_LOAD_WARNING)\n    }\n\n  }\n\n  if (root._bokeh_is_loading === 0) {\n    console.debug(\"Bokeh: BokehJS loaded, going straight to plotting\");\n    run_inline_js();\n  } else {\n    load_libs(css_urls, js_urls, function() {\n      console.debug(\"Bokeh: BokehJS plotting callback run at\", now());\n      run_inline_js();\n    });\n  }\n}(window));"
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "\n",
       "\n",
       "\n",
       "\n",
       "\n",
       "\n",
       "  <div class=\"bk-root\" id=\"aac6048b-5299-4ae5-a926-e5929be46bce\" data-root-id=\"1003\"></div>\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "application/javascript": [
       "(function(root) {\n",
       "  function embed_document(root) {\n",
       "    \n",
       "  var docs_json = {\"b159cf8c-939b-4e53-a6a5-51ca6403e73f\":{\"defs\":[],\"roots\":{\"references\":[{\"attributes\":{\"below\":[{\"id\":\"1014\"}],\"center\":[{\"id\":\"1017\"},{\"id\":\"1021\"},{\"id\":\"1051\"}],\"left\":[{\"id\":\"1018\"}],\"renderers\":[{\"id\":\"1039\"},{\"id\":\"1056\"}],\"title\":{\"id\":\"1004\"},\"toolbar\":{\"id\":\"1029\"},\"x_range\":{\"id\":\"1006\"},\"x_scale\":{\"id\":\"1010\"},\"y_range\":{\"id\":\"1008\"},\"y_scale\":{\"id\":\"1012\"}},\"id\":\"1003\",\"subtype\":\"Figure\",\"type\":\"Plot\"},{\"attributes\":{\"formatter\":{\"id\":\"1042\"},\"major_label_policy\":{\"id\":\"1044\"},\"ticker\":{\"id\":\"1019\"}},\"id\":\"1018\",\"type\":\"LinearAxis\"},{\"attributes\":{},\"id\":\"1010\",\"type\":\"LinearScale\"},{\"attributes\":{},\"id\":\"1012\",\"type\":\"LinearScale\"},{\"attributes\":{},\"id\":\"1027\",\"type\":\"HelpTool\"},{\"attributes\":{},\"id\":\"1008\",\"type\":\"DataRange1d\"},{\"attributes\":{},\"id\":\"1067\",\"type\":\"Selection\"},{\"attributes\":{},\"id\":\"1068\",\"type\":\"UnionRenderers\"},{\"attributes\":{},\"id\":\"1049\",\"type\":\"UnionRenderers\"},{\"attributes\":{\"formatter\":{\"id\":\"1045\"},\"major_label_policy\":{\"id\":\"1047\"},\"ticker\":{\"id\":\"1015\"}},\"id\":\"1014\",\"type\":\"LinearAxis\"},{\"attributes\":{},\"id\":\"1042\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{},\"id\":\"1015\",\"type\":\"BasicTicker\"},{\"attributes\":{\"axis\":{\"id\":\"1014\"},\"ticker\":null},\"id\":\"1017\",\"type\":\"Grid\"},{\"attributes\":{},\"id\":\"1044\",\"type\":\"AllLabels\"},{\"attributes\":{\"axis\":{\"id\":\"1018\"},\"dimension\":1,\"ticker\":null},\"id\":\"1021\",\"type\":\"Grid\"},{\"attributes\":{},\"id\":\"1019\",\"type\":\"BasicTicker\"},{\"attributes\":{\"line_color\":\"#0077bc\",\"line_width\":2,\"x\":{\"field\":\"x\"},\"y\":{\"field\":\"y\"}},\"id\":\"1037\",\"type\":\"Line\"},{\"attributes\":{\"active_multi\":null,\"tools\":[{\"id\":\"1022\"},{\"id\":\"1023\"},{\"id\":\"1024\"},{\"id\":\"1025\"},{\"id\":\"1026\"},{\"id\":\"1027\"}]},\"id\":\"1029\",\"type\":\"Toolbar\"},{\"attributes\":{},\"id\":\"1023\",\"type\":\"WheelZoomTool\"},{\"attributes\":{},\"id\":\"1022\",\"type\":\"PanTool\"},{\"attributes\":{\"overlay\":{\"id\":\"1028\"}},\"id\":\"1024\",\"type\":\"BoxZoomTool\"},{\"attributes\":{},\"id\":\"1045\",\"type\":\"BasicTickFormatter\"},{\"attributes\":{},\"id\":\"1025\",\"type\":\"SaveTool\"},{\"attributes\":{\"line_alpha\":0.1,\"line_color\":\"#0077bc\",\"line_width\":2,\"x\":{\"field\":\"x\"},\"y\":{\"field\":\"y\"}},\"id\":\"1038\",\"type\":\"Line\"},{\"attributes\":{},\"id\":\"1047\",\"type\":\"AllLabels\"},{\"attributes\":{},\"id\":\"1026\",\"type\":\"ResetTool\"},{\"attributes\":{\"label\":{\"value\":\"AUC = 0.781\"},\"renderers\":[{\"id\":\"1039\"}]},\"id\":\"1052\",\"type\":\"LegendItem\"},{\"attributes\":{\"line_color\":\"#d15555\",\"line_dash\":[2,4,6,4],\"line_width\":2,\"x\":{\"field\":\"x\"},\"y\":{\"field\":\"y\"}},\"id\":\"1054\",\"type\":\"Line\"},{\"attributes\":{\"data\":{\"x\":{\"__ndarray__\":\"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAlnsaYbmnoT+WexphuaehP2G5pxGWe6o/YbmnEZZ7qj+WexphuaexP5Z7GmG5p7E/fBphuacRtj98GmG5pxG2P5Z7GmG5p8E/lnsaYbmnwT8Jyz2NsNzDPwnLPY2w3MM/7mmE5Z5GyD/uaYTlnkbIP2G5pxGWe8o/YbmnEZZ7yj/UCMs9jbDMP9QIyz2NsMw/3dMIyz2N0D/d0wjLPY3QP08jLPc0wtI/TyMs9zTC0j81wnJPIyzXPzXCck8jLNc/7mmE5Z5G2D/uaYTlnkbYP6gRlnsaYdk/qBGWexph2T9huacRlnvaP2G5pxGWe9o/1AjLPY2w3D/UCMs9jbDcPwAAAAAAAOA/AAAAAAAA4D+NsNzTCMvtP42w3NMIy+0/AAAAAAAA8D8=\",\"dtype\":\"float64\",\"order\":\"little\",\"shape\":[40]},\"y\":{\"__ndarray__\":\"AAAAAAAAAAAAAAAAAACgPwAAAAAAAMA/AAAAAAAAwD8AAAAAAADIPwAAAAAAAMw/AAAAAAAA0D8AAAAAAADQPwAAAAAAANI/AAAAAAAA0j8AAAAAAADUPwAAAAAAANQ/AAAAAAAA3D8AAAAAAADcPwAAAAAAAN4/AAAAAAAA3j8AAAAAAADhPwAAAAAAAOE/AAAAAAAA5D8AAAAAAADkPwAAAAAAAOU/AAAAAAAA5T8AAAAAAADmPwAAAAAAAOY/AAAAAAAA5z8AAAAAAADnPwAAAAAAAOg/AAAAAAAA6D8AAAAAAADqPwAAAAAAAOo/AAAAAAAA7D8AAAAAAADsPwAAAAAAAO0/AAAAAAAA7T8AAAAAAADuPwAAAAAAAO4/AAAAAAAA7z8AAAAAAADvPwAAAAAAAPA/AAAAAAAA8D8=\",\"dtype\":\"float64\",\"order\":\"little\",\"shape\":[40]}},\"selected\":{\"id\":\"1048\"},\"selection_policy\":{\"id\":\"1049\"}},\"id\":\"1036\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"source\":{\"id\":\"1036\"}},\"id\":\"1040\",\"type\":\"CDSView\"},{\"attributes\":{\"items\":[{\"id\":\"1052\"}]},\"id\":\"1051\",\"type\":\"Legend\"},{\"attributes\":{\"source\":{\"id\":\"1053\"}},\"id\":\"1057\",\"type\":\"CDSView\"},{\"attributes\":{\"data\":{\"x\":[0,1],\"y\":[0,1]},\"selected\":{\"id\":\"1067\"},\"selection_policy\":{\"id\":\"1068\"}},\"id\":\"1053\",\"type\":\"ColumnDataSource\"},{\"attributes\":{\"line_alpha\":0.1,\"line_color\":\"#d15555\",\"line_dash\":[2,4,6,4],\"line_width\":2,\"x\":{\"field\":\"x\"},\"y\":{\"field\":\"y\"}},\"id\":\"1055\",\"type\":\"Line\"},{\"attributes\":{\"text\":\"ROC Curve - Test data\"},\"id\":\"1004\",\"type\":\"Title\"},{\"attributes\":{\"data_source\":{\"id\":\"1053\"},\"glyph\":{\"id\":\"1054\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1055\"},\"view\":{\"id\":\"1057\"}},\"id\":\"1056\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"1048\",\"type\":\"Selection\"},{\"attributes\":{\"data_source\":{\"id\":\"1036\"},\"glyph\":{\"id\":\"1037\"},\"hover_glyph\":null,\"muted_glyph\":null,\"nonselection_glyph\":{\"id\":\"1038\"},\"view\":{\"id\":\"1040\"}},\"id\":\"1039\",\"type\":\"GlyphRenderer\"},{\"attributes\":{},\"id\":\"1006\",\"type\":\"DataRange1d\"},{\"attributes\":{\"bottom_units\":\"screen\",\"fill_alpha\":0.5,\"fill_color\":\"lightgrey\",\"left_units\":\"screen\",\"level\":\"overlay\",\"line_alpha\":1.0,\"line_color\":\"black\",\"line_dash\":[4,4],\"line_width\":2,\"right_units\":\"screen\",\"syncable\":false,\"top_units\":\"screen\"},\"id\":\"1028\",\"type\":\"BoxAnnotation\"}],\"root_ids\":[\"1003\"]},\"title\":\"Bokeh Application\",\"version\":\"2.3.2\"}};\n",
       "  var render_items = [{\"docid\":\"b159cf8c-939b-4e53-a6a5-51ca6403e73f\",\"root_ids\":[\"1003\"],\"roots\":{\"1003\":\"aac6048b-5299-4ae5-a926-e5929be46bce\"}}];\n",
       "  root.Bokeh.embed.embed_items_notebook(docs_json, render_items);\n",
       "\n",
       "  }\n",
       "  if (root.Bokeh !== undefined) {\n",
       "    embed_document(root);\n",
       "  } else {\n",
       "    var attempts = 0;\n",
       "    var timer = setInterval(function(root) {\n",
       "      if (root.Bokeh !== undefined) {\n",
       "        clearInterval(timer);\n",
       "        embed_document(root);\n",
       "      } else {\n",
       "        attempts++;\n",
       "        if (attempts > 100) {\n",
       "          clearInterval(timer);\n",
       "          console.log(\"Bokeh: ERROR: Unable to run BokehJS code because BokehJS library is missing\");\n",
       "        }\n",
       "      }\n",
       "    }, 10, root)\n",
       "  }\n",
       "})(window);"
      ],
      "application/vnd.bokehjs_exec.v0+json": ""
     },
     "metadata": {
      "application/vnd.bokehjs_exec.v0+json": {
       "id": "1003"
      }
     },
     "output_type": "display_data"
    },
    {
     "data": {
      "image/png": "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\n",
      "text/plain": [
       "<Figure size 432x288 with 2 Axes>"
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
      "---------Performance on minority class on test data------------------\n",
      "\n",
      "Precision: 0.564103\n",
      "Recall: 0.687500\n",
      "F1 score: 0.619718\n"
     ]
    }
   ],
   "source": [
    "\n",
    "\n",
    "features=[ \n",
    "           'new_gender',\n",
    "           'new_CKD(t=0)',          \n",
    "            'new_Glucose_lastObs', 'new_Lipoprotein_lastObs', 'new_SBP_lastObs', 'new_HGB_lastObs'\n",
    "         ]\n",
    "\n",
    "# Inspect p values\n",
    "\n",
    "\n",
    "\n",
    "dframe_sm=dframe_outlierremoved_balanced[dframe_outlierremoved_balanced.index.isin(idx_0) | dframe_outlierremoved_balanced.index.isin(idx_1)]\n",
    "\n",
    "\n",
    "\n",
    "train, test = train_test_split(dframe_sm,test_size = 0.30,random_state=42)\n",
    "train = train.reset_index()\n",
    "test = test.reset_index()\n",
    "print ('\\n Training Records  and Test Records',[len(train),len(test)])\n",
    "print('\\n minority observation in test data')\n",
    "print(test[target].value_counts())\n",
    "print('\\n\\n')\n",
    "\n",
    "features_train = train[features]\n",
    "label_train = train[target]\n",
    "features_test = test[features]\n",
    "label_test = test[target]\n",
    "\n",
    "\n",
    "import statsmodels.api as sm\n",
    "logit_model=sm.Logit(label_train,features_train)\n",
    "result=logit_model.fit()\n",
    "print(result.summary())\n",
    "pred_test=result.predict(features_test)\n",
    "pred_test = list(map(round, pred_test))\n",
    "print('----------------Test data Performance-----------------------\\n')\n",
    "# precision tp / (tp + fp)\n",
    "precision = precision_score(label_test, pred_test )\n",
    "print('Precision: %f' % precision)\n",
    "# recall: tp / (tp + fn)\n",
    "recall = recall_score(label_test, pred_test )\n",
    "print('Recall: %f' % recall)\n",
    "# f1: 2 tp / (2 tp + fp + fn)\n",
    "f1 = f1_score(label_test, pred_test )\n",
    "print('F1 score: %f' % f1)\n",
    "\n",
    "\n",
    "\n",
    "# SKlearn\n",
    "classifier=my_clf.my_LogRegClassifier(dframe_outlierremoved_balanced,features,target)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "3ffccf53",
   "metadata": {},
   "source": [
    "#### <font color='green'> Feature Importance</font>"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "31b1739a",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[0.34275139 0.65311536 0.82747712 1.01319006 0.9153672  0.92286741]]\n",
      "\n",
      "\n"
     ]
    },
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
       "      <th>Feature</th>\n",
       "      <th>coeff</th>\n",
       "      <th>effect %</th>\n",
       "      <th>direction</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>new_Lipoprotein_lastObs</td>\n",
       "      <td>1.013190</td>\n",
       "      <td>0.216736</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>new_HGB_lastObs</td>\n",
       "      <td>0.922867</td>\n",
       "      <td>0.197415</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>new_SBP_lastObs</td>\n",
       "      <td>0.915367</td>\n",
       "      <td>0.195810</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>new_Glucose_lastObs</td>\n",
       "      <td>0.827477</td>\n",
       "      <td>0.177009</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>new_CKD(t=0)</td>\n",
       "      <td>0.653115</td>\n",
       "      <td>0.139711</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>5</th>\n",
       "      <td>new_gender</td>\n",
       "      <td>0.342751</td>\n",
       "      <td>0.073319</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "                   Feature     coeff  effect %   direction\n",
       "0  new_Lipoprotein_lastObs  1.013190   0.216736          1\n",
       "1          new_HGB_lastObs  0.922867   0.197415          1\n",
       "2          new_SBP_lastObs  0.915367   0.195810          1\n",
       "3      new_Glucose_lastObs  0.827477   0.177009          1\n",
       "4             new_CKD(t=0)  0.653115   0.139711          1\n",
       "5               new_gender  0.342751   0.073319          1"
      ]
     },
     "execution_count": 24,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "print(classifier.coef_)\n",
    "print('\\n')\n",
    "dframe_coff=pd.DataFrame()\n",
    "dframe_coff['Feature']=features\n",
    "dframe_coff['coeff']=classifier.coef_[0]\n",
    "dframe_coff['effect % ']=dframe_coff['coeff'].apply(lambda x: abs(x)/sum(abs(dframe_coff['coeff'])))\n",
    "dframe_coff['direction']=dframe_coff['coeff'].apply(lambda x: +1 if x>0 else 0 if x==0 else -1)\n",
    "from matplotlib.pyplot import figure\n",
    "\n",
    "\n",
    "plt.bar(dframe_coff['Feature'],dframe_coff['coeff'])\n",
    "plt.xticks(range(len(features)),features, rotation=90)\n",
    "plt.show()\n",
    "\n",
    "dframe_coff.sort_values('effect % ',ascending=False).reset_index(drop=True).head(20)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 25,
   "id": "cd83b382",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Fitting 2 folds for each of 10 candidates, totalling 20 fits\n",
      "\n",
      "\n",
      "\n",
      " Accuracy Train ------- Accuracy Test------------ AUC Train-------AUC Test--------\n",
      "0.8047619047619048 0.6666666666666666 0.8679059652029827 0.6928879310344828\n",
      "Precision: 0.527778\n",
      "Recall: 0.593750\n"
     ]
    },
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
    "features=[ \n",
    "           'new_gender',\n",
    "           'new_CKD(t=0)',          \n",
    "            'new_Glucose_lastObs', 'new_Lipoprotein_lastObs', 'new_SBP_lastObs', 'new_HGB_lastObs'\n",
    "         ]\n",
    "         \n",
    "classifier=my_clf.my_RFClassifier(dframe_outlierremoved_balanced,features,target)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "e8995885",
   "metadata": {},
   "source": [
    "#### <font color='grey'> Advanced Approach(Deep Learning)</font>"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "f4e0e3d8",
   "metadata": {},
   "source": [
    "The recall value of the linear model(s) are quite poor. Let us try to improve it using the identified features but with the longitudinal data nature intact. We will build a RNN. But before that we need to standardise the observation period across patients to some extent. By visual inspection of the file 'Window.csv' produced by the code we can come to a cutoff period."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 26,
   "id": "1159acf8",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "(69, 1)\n",
      "(300, 1)\n",
      "   id  Time Window\n",
      "0   0            0\n",
      "1   0            1\n",
      "2   0            2\n",
      "3   0            3\n",
      "4   0            4\n",
      "(20700, 2)\n"
     ]
    }
   ],
   "source": [
    "\n",
    "dframe_crossjoin1=pd.DataFrame(dframe_smoothened_test_results[\"Time Window\"].sort_values().unique())\n",
    "dframe_crossjoin2=pd.DataFrame(dframe_smoothened_test_results[\"id\"].sort_values().unique())\n",
    "print(dframe_crossjoin1.shape)\n",
    "print(dframe_crossjoin2.shape)\n",
    "dframe_cros12=dframe_crossjoin2.merge(dframe_crossjoin1,how='cross')\n",
    "dframe_cros12.columns=['id','Time Window']\n",
    "print(dframe_cros12.head(5))\n",
    "\n",
    "print(dframe_cros12.shape)\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "b664e10a",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "(20700, 20)"
      ]
     },
     "execution_count": 27,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dframe_temporal=dframe_smoothened_test_results.merge(dframe_cros12,how='right',on=['id','Time Window'])\n",
    "dframe_temporal.update(dframe_temporal.sort_values([\"id\",'Time Window']).groupby(\"id\").ffill())\n",
    "dframe_temporal=pd.DataFrame(dframe_temporal.groupby(['id','Time Window'],as_index=False).max())\n",
    "# dframe_temporal=dframe_demographics[['id','gender']].merge(dframe_temporal,on='id',how='inner')\n",
    "dframe_temporal.to_csv('output/Normalized_Window.csv')\n",
    "\n",
    "\n",
    "dframe_temporal.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "54d7870f",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "\n",
      " Original shape of dataframe-----> (20700, 20)\n",
      "id                    int64\n",
      "Time Window           int64\n",
      "Creatinine_min      float64\n",
      "DBP_min             float64\n",
      "SBP_min             float64\n",
      "Glucose_min         float64\n",
      "HGB_min             float64\n",
      "Lipoprotein_min     float64\n",
      "Creatinine_mean     float64\n",
      "DBP_mean            float64\n",
      "SBP_mean            float64\n",
      "Glucose_mean        float64\n",
      "HGB_mean            float64\n",
      "Lipoprotein_mean    float64\n",
      "Creatinine_max      float64\n",
      "DBP_max             float64\n",
      "SBP_max             float64\n",
      "Glucose_max         float64\n",
      "HGB_max             float64\n",
      "Lipoprotein_max     float64\n",
      "dtype: object\n",
      "Your selected dataframe has 20 columns.\n",
      "There are 0 columns that have missing values.\n",
      "\n",
      " ............................  Initiating feature scaling.....\n",
      "\n",
      "\n",
      "Columns that are numerics----> 18\n",
      "\n",
      " --------------- Scaling of features completed with new dataframe size---------> (20700, 18)\n",
      "\n",
      "--------------------Initiating categorical encoding--------------------------------\n",
      "\n",
      "\n",
      "The columns to be encoded------- Index([], dtype='object')\n",
      "\n",
      "------------Completed Scaling and Encoding-----------------------\n",
      "\n",
      "\n",
      " Final dataframe size\n",
      " (20700, 20)\n",
      "Creatinine_max      float64\n",
      "Creatinine_mean     float64\n",
      "Creatinine_min      float64\n",
      "DBP_max             float64\n",
      "DBP_mean            float64\n",
      "DBP_min             float64\n",
      "Glucose_max         float64\n",
      "Glucose_mean        float64\n",
      "Glucose_min         float64\n",
      "HGB_max             float64\n",
      "HGB_mean            float64\n",
      "HGB_min             float64\n",
      "Lipoprotein_max     float64\n",
      "Lipoprotein_mean    float64\n",
      "Lipoprotein_min     float64\n",
      "SBP_max             float64\n",
      "SBP_mean            float64\n",
      "SBP_min             float64\n",
      "id                    int64\n",
      "Time Window           int64\n",
      "dtype: object\n",
      "Your selected dataframe has 20 columns.\n",
      "There are 0 columns that have missing values.\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "(20700, 20)"
      ]
     },
     "execution_count": 28,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "dframe_temporal_scaled=my_internal_func.my_preprocess(dframe_temporal,'standard',ignore=['id','Time Window'])\n",
    "dframe_temporal_scaled.to_csv('output/Scaled_Normalized_Window.csv',index=False)\n",
    "dframe_temporal_scaled.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "2f1a6572",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "69 6\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "Index(['Creatinine_max', 'DBP_max', 'Glucose_max', 'HGB_max',\n",
       "       'Lipoprotein_max', 'SBP_max'],\n",
       "      dtype='object')"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "features=dframe_temporal_scaled.filter(like='_max',axis=1).columns.difference(['id','Time Window'])\n",
    "len_features=len(features)\n",
    "time_window=len(dframe_temporal_scaled['Time Window'].unique())\n",
    "print(time_window,len_features)\n",
    "features\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "708cda1d",
   "metadata": {},
   "outputs": [],
   "source": [
    " def my_reshape(df,time_window,len_features):\n",
    "        # Getting unique  ids.\n",
    "        Series_ids = df['id'].unique()\n",
    "        print(len(Series_ids))\n",
    "        # Final array having the result.\n",
    "        final = np.zeros((Series_ids.shape[0], time_window, len_features))\n",
    "        idx=0\n",
    "        for ids in Series_ids:\n",
    "            dframe_=df[df.id==ids][features]\n",
    "            final[idx]=dframe_.values\n",
    "        #     print(final[idx])\n",
    "        #     print('\\n============================')\n",
    "        return final\n",
    "    "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "83668ad0",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "150 150\n",
      "(10350, 20) (10350, 20)\n"
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
       "      <th>Creatinine_max</th>\n",
       "      <th>Creatinine_mean</th>\n",
       "      <th>Creatinine_min</th>\n",
       "      <th>DBP_max</th>\n",
       "      <th>DBP_mean</th>\n",
       "      <th>DBP_min</th>\n",
       "      <th>Glucose_max</th>\n",
       "      <th>Glucose_mean</th>\n",
       "      <th>Glucose_min</th>\n",
       "      <th>HGB_max</th>\n",
       "      <th>HGB_mean</th>\n",
       "      <th>HGB_min</th>\n",
       "      <th>Lipoprotein_max</th>\n",
       "      <th>Lipoprotein_mean</th>\n",
       "      <th>Lipoprotein_min</th>\n",
       "      <th>SBP_max</th>\n",
       "      <th>SBP_mean</th>\n",
       "      <th>SBP_min</th>\n",
       "      <th>id</th>\n",
       "      <th>Time Window</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>138</th>\n",
       "      <td>3.368019</td>\n",
       "      <td>3.376968</td>\n",
       "      <td>3.376967</td>\n",
       "      <td>0.156049</td>\n",
       "      <td>0.164318</td>\n",
       "      <td>0.172134</td>\n",
       "      <td>0.330319</td>\n",
       "      <td>0.349396</td>\n",
       "      <td>0.366972</td>\n",
       "      <td>0.844719</td>\n",
       "      <td>0.848976</td>\n",
       "      <td>0.852512</td>\n",
       "      <td>-0.892484</td>\n",
       "      <td>-0.890905</td>\n",
       "      <td>-0.888904</td>\n",
       "      <td>-0.032206</td>\n",
       "      <td>-0.025271</td>\n",
       "      <td>-0.018088</td>\n",
       "      <td>2</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>139</th>\n",
       "      <td>3.368019</td>\n",
       "      <td>3.376968</td>\n",
       "      <td>3.376967</td>\n",
       "      <td>0.156049</td>\n",
       "      <td>0.164318</td>\n",
       "      <td>0.172134</td>\n",
       "      <td>0.330319</td>\n",
       "      <td>0.349396</td>\n",
       "      <td>0.366972</td>\n",
       "      <td>0.844719</td>\n",
       "      <td>0.848976</td>\n",
       "      <td>0.852512</td>\n",
       "      <td>-0.892484</td>\n",
       "      <td>-0.890905</td>\n",
       "      <td>-0.888904</td>\n",
       "      <td>-0.032206</td>\n",
       "      <td>-0.025271</td>\n",
       "      <td>-0.018088</td>\n",
       "      <td>2</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>140</th>\n",
       "      <td>3.368019</td>\n",
       "      <td>3.376968</td>\n",
       "      <td>3.376967</td>\n",
       "      <td>0.156049</td>\n",
       "      <td>0.164318</td>\n",
       "      <td>0.172134</td>\n",
       "      <td>0.330319</td>\n",
       "      <td>0.349396</td>\n",
       "      <td>0.366972</td>\n",
       "      <td>0.844719</td>\n",
       "      <td>0.848976</td>\n",
       "      <td>0.852512</td>\n",
       "      <td>-0.892484</td>\n",
       "      <td>-0.890905</td>\n",
       "      <td>-0.888904</td>\n",
       "      <td>-0.032206</td>\n",
       "      <td>-0.025271</td>\n",
       "      <td>-0.018088</td>\n",
       "      <td>2</td>\n",
       "      <td>2</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>141</th>\n",
       "      <td>3.368019</td>\n",
       "      <td>3.376968</td>\n",
       "      <td>3.376967</td>\n",
       "      <td>0.156049</td>\n",
       "      <td>0.164318</td>\n",
       "      <td>0.172134</td>\n",
       "      <td>0.330319</td>\n",
       "      <td>0.349396</td>\n",
       "      <td>0.366972</td>\n",
       "      <td>0.844719</td>\n",
       "      <td>0.848976</td>\n",
       "      <td>0.852512</td>\n",
       "      <td>-0.892484</td>\n",
       "      <td>-0.890905</td>\n",
       "      <td>-0.888904</td>\n",
       "      <td>-0.032206</td>\n",
       "      <td>-0.025271</td>\n",
       "      <td>-0.018088</td>\n",
       "      <td>2</td>\n",
       "      <td>3</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>142</th>\n",
       "      <td>3.368019</td>\n",
       "      <td>3.376968</td>\n",
       "      <td>3.376967</td>\n",
       "      <td>0.156049</td>\n",
       "      <td>0.164318</td>\n",
       "      <td>0.172134</td>\n",
       "      <td>0.330319</td>\n",
       "      <td>0.349396</td>\n",
       "      <td>0.366972</td>\n",
       "      <td>0.844719</td>\n",
       "      <td>0.848976</td>\n",
       "      <td>0.852512</td>\n",
       "      <td>-0.892484</td>\n",
       "      <td>-0.890905</td>\n",
       "      <td>-0.888904</td>\n",
       "      <td>-0.032206</td>\n",
       "      <td>-0.025271</td>\n",
       "      <td>-0.018088</td>\n",
       "      <td>2</td>\n",
       "      <td>4</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "</div>"
      ],
      "text/plain": [
       "     Creatinine_max  Creatinine_mean  Creatinine_min   DBP_max  DBP_mean  \\\n",
       "138        3.368019         3.376968        3.376967  0.156049  0.164318   \n",
       "139        3.368019         3.376968        3.376967  0.156049  0.164318   \n",
       "140        3.368019         3.376968        3.376967  0.156049  0.164318   \n",
       "141        3.368019         3.376968        3.376967  0.156049  0.164318   \n",
       "142        3.368019         3.376968        3.376967  0.156049  0.164318   \n",
       "\n",
       "      DBP_min  Glucose_max  Glucose_mean  Glucose_min   HGB_max  HGB_mean  \\\n",
       "138  0.172134     0.330319      0.349396     0.366972  0.844719  0.848976   \n",
       "139  0.172134     0.330319      0.349396     0.366972  0.844719  0.848976   \n",
       "140  0.172134     0.330319      0.349396     0.366972  0.844719  0.848976   \n",
       "141  0.172134     0.330319      0.349396     0.366972  0.844719  0.848976   \n",
       "142  0.172134     0.330319      0.349396     0.366972  0.844719  0.848976   \n",
       "\n",
       "      HGB_min  Lipoprotein_max  Lipoprotein_mean  Lipoprotein_min   SBP_max  \\\n",
       "138  0.852512        -0.892484         -0.890905        -0.888904 -0.032206   \n",
       "139  0.852512        -0.892484         -0.890905        -0.888904 -0.032206   \n",
       "140  0.852512        -0.892484         -0.890905        -0.888904 -0.032206   \n",
       "141  0.852512        -0.892484         -0.890905        -0.888904 -0.032206   \n",
       "142  0.852512        -0.892484         -0.890905        -0.888904 -0.032206   \n",
       "\n",
       "     SBP_mean   SBP_min  id  Time Window  \n",
       "138 -0.025271 -0.018088   2            0  \n",
       "139 -0.025271 -0.018088   2            1  \n",
       "140 -0.025271 -0.018088   2            2  \n",
       "141 -0.025271 -0.018088   2            3  \n",
       "142 -0.025271 -0.018088   2            4  "
      ]
     },
     "execution_count": 31,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "train_idx=list(dframe_final[dframe_final['target']==0]['id'].sample(100).unique())\n",
    "train_idx.extend(list(dframe_final[dframe_final['target']==1]['id'].sample(50).unique()))\n",
    "test_idx=dframe_final['id'][~dframe_final['id'].isin(train_idx)]\n",
    "print(len(train_idx),len(test_idx))\n",
    "dframe_X_train=dframe_temporal_scaled[dframe_temporal_scaled.id.isin(train_idx)]\n",
    "dframe_X_test=dframe_temporal_scaled[dframe_temporal_scaled.id.isin(test_idx)]\n",
    "# print(dframe_X_train.target.value_counts())\n",
    "# print(dframe_X_test.target.value_counts())\n",
    "\n",
    "dframe_Y_train=np.array(dframe_final[dframe_final.id.isin(train_idx)]['target']).astype(float)\n",
    "dframe_Y_test=np.array(dframe_final[dframe_final.id.isin(test_idx)]['target']).astype(float)\n",
    "print(dframe_X_train.shape,dframe_X_test.shape)\n",
    "\n",
    "dframe_X_aux_train=np.array(dframe_final['gender'].apply(lambda x: 1 if x=='Male'else 0)[dframe_final.id.isin(train_idx)]).astype(int)\n",
    "dframe_X_aux_test=np.array(dframe_final['gender'].apply(lambda x: 1 if x=='Male'else 0)[dframe_final.id.isin(test_idx)]).astype(int)\n",
    "dframe_X_train.head()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 32,
   "id": "358e3e80",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "150\n",
      "150\n"
     ]
    },
    {
     "data": {
      "text/plain": [
       "((150, 69, 6), (150, 69, 6))"
      ]
     },
     "execution_count": 32,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "# # Getting unique  ids.\n",
    "# Series_ids = dframe_temporal_scaled['id'].unique()\n",
    "# print(len(Series_ids))\n",
    "# # Final array having the result.\n",
    "# final = np.zeros((Series_ids.shape[0], time_window, len_features))\n",
    "# idx=0\n",
    "# for ids in Series_ids:\n",
    "#     dframe_=dframe_temporal_scaled[dframe_temporal_scaled.index.values==ids][features]\n",
    "#     final[idx]=dframe_.values\n",
    "# #     print(final[idx])\n",
    "# #     print('\\n============================')\n",
    "    \n",
    "    \n",
    "    \n",
    "tensor_X_train= my_reshape(dframe_X_train,time_window,len_features) \n",
    "tensor_X_test= my_reshape(dframe_X_test,time_window,len_features) \n",
    "tensor_X_train.shape,tensor_X_test.shape"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "26330ba8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Import `Sequential` from `keras.models`\n",
    "from keras.models import Sequential\n",
    "# Import `Dense` from `keras.layers`\n",
    "from keras.layers import Dense, Dropout\n",
    "from keras.layers import LSTM\n",
    "from keras.layers import Input\n",
    "from keras.models import Model\n",
    "import keras\n",
    "\n",
    "\n",
    "# main_input = Input(shape=(tensor_X_train.shape[1], tensor_X_train.shape[2]), name='main_input')\n",
    "# lstm_out = LSTM(256, dropout=0.1, recurrent_dropout=0.1)(main_input)\n",
    "\n",
    "\n",
    "# auxiliary_output = Dense(10, activation='relu', name='aux_output')(lstm_out)\n",
    "\n",
    "# auxiliary_input = Input(shape=(1,), name='aux_input')\n",
    "# x = keras.layers.concatenate([lstm_out, auxiliary_input])\n",
    "\n",
    "\n",
    "# # And tensor_X_trainly we add the main logistic regression layer\n",
    "# main_output = Dense(1, activation='sigmoid', name='main_output')(x)\n",
    "\n",
    "# #This defines a model with two inputs and two outputs:\n",
    "# model = Model(inputs=[main_input, auxiliary_input], outputs=[main_output, auxiliary_output])\n",
    "\n",
    "\n",
    "model = Sequential()\n",
    "# define first hidden layer and visible layer\n",
    "model.add(LSTM(100,return_sequences=True,input_shape=(tensor_X_train.shape[1], tensor_X_train.shape[2]),name='main_input'))\n",
    "# define output layer\n",
    "model.add(Dense(1, activation='sigmoid',name='main_output'))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "43b2c67d",
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
      " main_input (LSTM)           (None, 69, 100)           42800     \n",
      "                                                                 \n",
      " main_output (Dense)         (None, 69, 1)             101       \n",
      "                                                                 \n",
      "=================================================================\n",
      "Total params: 42,901\n",
      "Trainable params: 42,901\n",
      "Non-trainable params: 0\n",
      "_________________________________________________________________\n"
     ]
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "9648129b",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "1c2c9c0f",
   "metadata": {},
   "outputs": [],
   "source": [
    "from keras import backend as K\n",
    "\n",
    "def recall_m(y_true, y_pred):\n",
    "    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))\n",
    "    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))\n",
    "    recall = true_positives / (possible_positives + K.epsilon())\n",
    "    return recall\n",
    "\n",
    "def precision_m(y_true, y_pred):\n",
    "    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))\n",
    "    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))\n",
    "    precision = true_positives / (predicted_positives + K.epsilon())\n",
    "    return precision\n",
    "\n",
    "def f1_m(y_true, y_pred):\n",
    "    precision = precision_m(y_true, y_pred)\n",
    "    recall = recall_m(y_true, y_pred)\n",
    "    return 2*((precision*recall)/(precision+recall+K.epsilon()))\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "d27b47e6",
   "metadata": {},
   "outputs": [],
   "source": [
    "class_weight = {0 : 1.0, 1: 50.0}\n",
    "model.compile( optimizer='adam',\n",
    "               loss= 'binary_crossentropy',\n",
    "               metrics=['acc',f1_m,precision_m, recall_m])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "c495be92",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/200\n"
     ]
    },
    {
     "ename": "ValueError",
     "evalue": "in user code:\n\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 1021, in train_function  *\n        return step_function(self, iterator)\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 1010, in step_function  **\n        outputs = model.distribute_strategy.run(run_step, args=(data,))\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 1000, in run_step  **\n        outputs = model.train_step(data)\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 859, in train_step\n        y_pred = self(x, training=True)\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\utils\\traceback_utils.py\", line 67, in error_handler\n        raise e.with_traceback(filtered_tb) from None\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\input_spec.py\", line 183, in assert_input_compatibility\n        raise ValueError(f'Missing data for input \"{name}\". '\n\n    ValueError: Missing data for input \"main_input_input\". You passed a data dictionary with keys ['main_input']. Expected the following keys: ['main_input_input']\n",
     "output_type": "error",
     "traceback": [
      "\u001b[1;31m---------------------------------------------------------------------------\u001b[0m",
      "\u001b[1;31mValueError\u001b[0m                                Traceback (most recent call last)",
      "\u001b[1;32m<ipython-input-39-950a70f4d270>\u001b[0m in \u001b[0;36m<module>\u001b[1;34m\u001b[0m\n\u001b[1;32m----> 1\u001b[1;33m history=model.fit({'main_input': tensor_X_train},\n\u001b[0m\u001b[0;32m      2\u001b[0m           \u001b[1;33m{\u001b[0m\u001b[1;34m'main_output'\u001b[0m\u001b[1;33m:\u001b[0m\u001b[0mdframe_Y_train\u001b[0m\u001b[1;33m}\u001b[0m\u001b[1;33m,\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m      3\u001b[0m           epochs=200, batch_size=128,validation_split=0.11)\n",
      "\u001b[1;32mC:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\utils\\traceback_utils.py\u001b[0m in \u001b[0;36merror_handler\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m     65\u001b[0m     \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m  \u001b[1;31m# pylint: disable=broad-except\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     66\u001b[0m       \u001b[0mfiltered_tb\u001b[0m \u001b[1;33m=\u001b[0m \u001b[0m_process_traceback_frames\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0m__traceback__\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m---> 67\u001b[1;33m       \u001b[1;32mraise\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mwith_traceback\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0mfiltered_tb\u001b[0m\u001b[1;33m)\u001b[0m \u001b[1;32mfrom\u001b[0m \u001b[1;32mNone\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m     68\u001b[0m     \u001b[1;32mfinally\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m     69\u001b[0m       \u001b[1;32mdel\u001b[0m \u001b[0mfiltered_tb\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;32m~\\AppData\\Roaming\\Python\\Python38\\site-packages\\tensorflow\\python\\framework\\func_graph.py\u001b[0m in \u001b[0;36mautograph_handler\u001b[1;34m(*args, **kwargs)\u001b[0m\n\u001b[0;32m   1145\u001b[0m           \u001b[1;32mexcept\u001b[0m \u001b[0mException\u001b[0m \u001b[1;32mas\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m:\u001b[0m  \u001b[1;31m# pylint:disable=broad-except\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1146\u001b[0m             \u001b[1;32mif\u001b[0m \u001b[0mhasattr\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m,\u001b[0m \u001b[1;34m\"ag_error_metadata\"\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[1;32m-> 1147\u001b[1;33m               \u001b[1;32mraise\u001b[0m \u001b[0me\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mag_error_metadata\u001b[0m\u001b[1;33m.\u001b[0m\u001b[0mto_exception\u001b[0m\u001b[1;33m(\u001b[0m\u001b[0me\u001b[0m\u001b[1;33m)\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0m\u001b[0;32m   1148\u001b[0m             \u001b[1;32melse\u001b[0m\u001b[1;33m:\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n\u001b[0;32m   1149\u001b[0m               \u001b[1;32mraise\u001b[0m\u001b[1;33m\u001b[0m\u001b[1;33m\u001b[0m\u001b[0m\n",
      "\u001b[1;31mValueError\u001b[0m: in user code:\n\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 1021, in train_function  *\n        return step_function(self, iterator)\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 1010, in step_function  **\n        outputs = model.distribute_strategy.run(run_step, args=(data,))\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 1000, in run_step  **\n        outputs = model.train_step(data)\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\training.py\", line 859, in train_step\n        y_pred = self(x, training=True)\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\utils\\traceback_utils.py\", line 67, in error_handler\n        raise e.with_traceback(filtered_tb) from None\n    File \"C:\\ProgramData\\Anaconda3\\lib\\site-packages\\keras\\engine\\input_spec.py\", line 183, in assert_input_compatibility\n        raise ValueError(f'Missing data for input \"{name}\". '\n\n    ValueError: Missing data for input \"main_input_input\". You passed a data dictionary with keys ['main_input']. Expected the following keys: ['main_input_input']\n"
     ]
    }
   ],
   "source": [
    "history=model.fit({'main_input': tensor_X_train},\n",
    "          {'main_output':dframe_Y_train},\n",
    "          epochs=200, batch_size=128,validation_split=0.11)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5937111d",
   "metadata": {},
   "outputs": [],
   "source": [
    "# summarize history for accuracy\n",
    "plt.plot(history.history['main_output_acc'])\n",
    "plt.plot(history.history['val_main_output_acc'])\n",
    "plt.title('main model acc')\n",
    "plt.ylabel('acc')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'Validation'], loc='upper left')\n",
    "plt.show()\n",
    "# summarize history for loss\n",
    "plt.plot(history.history['main_output_loss'])\n",
    "plt.plot(history.history['val_main_output_loss'])\n",
    "plt.title('main model loss')\n",
    "plt.ylabel('loss')\n",
    "plt.xlabel('epoch')\n",
    "plt.legend(['train', 'Validation'], loc='upper left')\n",
    "plt.show()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "4334abe4",
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
    "    plt.show()\n",
    "\n",
    "\n",
    "\n",
    "\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "3d025c37",
   "metadata": {},
   "outputs": [],
   "source": [
    "yhat = model.predict({'main_input': tensor_X_test, 'aux_input': dframe_X_aux_test})\n",
    "compute_roc(dframe_Y_test,yhat[0],'deep learning')\n",
    "compute_roc(dframe_Y_test,yhat[0],'deep learning')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "17a6a290",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.metrics import classification_report\n",
    "print(classification_report(dframe_Y_test, (yhat[0]>0.5).astype(int)))"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "ece149b9",
   "metadata": {},
   "source": [
    "### Conclusions"
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
################################################################################
