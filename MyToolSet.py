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
       "<p>300 rows × 60 columns</p>\n",
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
       "<p>300 rows × 60 columns</p>\n",
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
      "(150, 60) (150, 60)\n",
      "(150, 56)\n",
      "(150, 56)\n"
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
       "      <th>1</th>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.915973</td>\n",
       "      <td>1.903572</td>\n",
       "      <td>1.903572</td>\n",
       "      <td>1.903572</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.476681</td>\n",
       "      <td>-0.476681</td>\n",
       "      <td>-0.476681</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.269572</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>-0.536878</td>\n",
       "      <td>0.436677</td>\n",
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
       "      <th>12</th>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>-0.105438</td>\n",
       "      <td>0.378213</td>\n",
       "      <td>...</td>\n",
       "      <td>-0.472022</td>\n",
       "      <td>-0.472022</td>\n",
       "      <td>-0.990870</td>\n",
       "      <td>-0.990870</td>\n",
       "      <td>-0.990870</td>\n",
       "      <td>-0.990870</td>\n",
       "      <td>-0.990870</td>\n",
       "      <td>-0.842884</td>\n",
       "      <td>-0.842884</td>\n",
       "      <td>-0.842884</td>\n",
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
       "      <th>290</th>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.725449</td>\n",
       "      <td>0.483624</td>\n",
       "      <td>0.142588</td>\n",
       "      <td>0.123986</td>\n",
       "      <td>0.123986</td>\n",
       "      <td>1.438524</td>\n",
       "      <td>1.438524</td>\n",
       "      <td>...</td>\n",
       "      <td>-1.156772</td>\n",
       "      <td>-1.230944</td>\n",
       "      <td>-1.230944</td>\n",
       "      <td>-0.972954</td>\n",
       "      <td>-0.972954</td>\n",
       "      <td>-0.972954</td>\n",
       "      <td>-1.460986</td>\n",
       "      <td>-0.500330</td>\n",
       "      <td>-0.500330</td>\n",
       "      <td>-1.156772</td>\n",
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
       "<p>150 rows × 56 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Glucose_1.0  Glucose_2.0  Glucose_3.0  Glucose_4.0  Glucose_5.0  \\\n",
       "0      -0.564285    -0.564285    -0.564285    -0.564285    -0.564285   \n",
       "1       1.915973     1.915973     1.915973     1.915973     1.915973   \n",
       "3      -0.570486    -0.570486    -0.570486    -0.570486    -0.570486   \n",
       "6      -0.012428    -0.012428    -0.012428    -0.012428    -0.012428   \n",
       "12     -0.105438    -0.105438    -0.105438    -0.105438    -0.105438   \n",
       "..           ...          ...          ...          ...          ...   \n",
       "290     0.725449     0.725449     0.725449     0.725449     0.483624   \n",
       "292     0.861863     0.861863     0.861863     0.861863     0.861863   \n",
       "295    -0.520881    -0.520881    -0.520881    -0.520881    -0.266654   \n",
       "298    -0.440273    -0.440273    -0.440273    -0.837114    -0.837114   \n",
       "299    -0.303858    -0.303858    -0.303858    -0.341062    -0.341062   \n",
       "\n",
       "     Glucose_6.0  Glucose_7.0  Glucose_8.0  Glucose_9.0  Glucose_10.0  ...  \\\n",
       "0      -0.663496    -0.663496    -0.663496     0.123986      0.123986  ...   \n",
       "1       1.915973     1.915973     1.903572     1.903572      1.903572  ...   \n",
       "3      -0.570486    -0.570486    -0.793709     0.458821      0.458821  ...   \n",
       "6      -0.012428     0.465022     0.465022     0.465022      1.345514  ...   \n",
       "12     -0.105438    -0.105438    -0.105438    -0.105438      0.378213  ...   \n",
       "..           ...          ...          ...          ...           ...  ...   \n",
       "290     0.142588     0.123986     0.123986     1.438524      1.438524  ...   \n",
       "292     0.644841     0.644841     0.644841     0.644841      0.644841  ...   \n",
       "295    -0.266654    -0.961127    -0.961127    -0.961127     -0.961127  ...   \n",
       "298    -0.837114    -0.781308    -0.781308    -0.781308     -0.272855  ...   \n",
       "299    -0.341062    -0.341062    -0.341062    -0.675897     -0.675897  ...   \n",
       "\n",
       "     Lipoprotein_5.0  Lipoprotein_6.0  Lipoprotein_7.0  Lipoprotein_8.0  \\\n",
       "0           2.556857         2.556857         2.556857         2.556857   \n",
       "1          -0.476681        -0.476681        -0.476681        -0.269572   \n",
       "3           0.533065         0.533065         0.533065         0.533065   \n",
       "6          -1.745133        -1.745133        -1.745133        -1.360298   \n",
       "12         -0.472022        -0.472022        -0.990870        -0.990870   \n",
       "..               ...              ...              ...              ...   \n",
       "290        -1.156772        -1.230944        -1.230944        -0.972954   \n",
       "292         0.221685         0.153963         0.153963         0.153963   \n",
       "295         1.333194         0.746266         0.746266         0.746266   \n",
       "298        -0.356643        -0.356643         0.597205         0.597205   \n",
       "299        -0.475247        -0.475247        -0.719980        -0.719980   \n",
       "\n",
       "     Lipoprotein_9.0  Lipoprotein_10.0  Lipoprotein_11.0  Lipoprotein_12.0  \\\n",
       "0           2.556857          2.556857          0.890310          0.890310   \n",
       "1          -0.269572         -0.269572         -0.536878         -0.536878   \n",
       "3           0.533065          0.533065         -0.205791         -0.205791   \n",
       "6          -1.360298         -1.360298         -1.454894         -1.454894   \n",
       "12         -0.990870         -0.990870         -0.990870         -0.842884   \n",
       "..               ...               ...               ...               ...   \n",
       "290        -0.972954         -0.972954         -1.460986         -0.500330   \n",
       "292        -0.021614         -0.021614          0.441694          0.441694   \n",
       "295         0.746266          0.746266          0.746266          0.746266   \n",
       "298         0.597205          0.597205          0.597205          0.597205   \n",
       "299        -0.719980         -0.719980         -0.719980         -0.719980   \n",
       "\n",
       "     Lipoprotein_13.0  Lipoprotein_14.0  \n",
       "0            0.890310          0.890310  \n",
       "1           -0.536878          0.436677  \n",
       "3           -0.205791          0.850895  \n",
       "6           -1.454894         -1.243127  \n",
       "12          -0.842884         -0.842884  \n",
       "..                ...               ...  \n",
       "290         -0.500330         -1.156772  \n",
       "292          0.441694          0.441694  \n",
       "295          1.252930          1.252930  \n",
       "298          0.597205         -0.613200  \n",
       "299         -0.719980         -0.719980  \n",
       "\n",
       "[150 rows x 56 columns]"
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
    "minority_test_id=pd.Series(minority_id).sample(frac=0.5,random_state=0)\n",
    "majority_test_id=pd.Series(majority_id).sample(frac=0.5,random_state=0)\n",
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
    "import keras"
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
       "(150, 14, 4)"
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
    "\n",
    "main_input = Input(shape=(dframe_train_X_time.shape[1], dframe_train_X_time.shape[2]), name='main_input')\n",
    "\n",
    "lstm_out = LSTM(16, dropout=0.25, recurrent_dropout=0.25)(main_input)\n",
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
    "              loss={'main_output': 'binary_crossentropy', 'aux_output': 'binary_crossentropy'},\n",
    "              loss_weights={'main_output': 1., 'aux_output': 0.5},metrics=['acc'])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5c730879",
   "metadata": {},
   "outputs": [],
   "source": []
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b289e096",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Epoch 1/300\n",
      "5/5 [==============================] - 3s 119ms/step - loss: 1.0900 - main_output_loss: 0.7456 - aux_output_loss: 0.6887 - main_output_acc: 0.3333 - aux_output_acc: 0.5481 - val_loss: 1.0938 - val_main_output_loss: 0.7364 - val_aux_output_loss: 0.7148 - val_main_output_acc: 0.3333 - val_aux_output_acc: 0.4000\n",
      "Epoch 2/300\n",
      "5/5 [==============================] - 0s 12ms/step - loss: 1.0817 - main_output_loss: 0.7406 - aux_output_loss: 0.6822 - main_output_acc: 0.3704 - aux_output_acc: 0.4963 - val_loss: 1.0790 - val_main_output_loss: 0.7276 - val_aux_output_loss: 0.7029 - val_main_output_acc: 0.3333 - val_aux_output_acc: 0.3333\n",
      "Epoch 3/300\n",
      "5/5 [==============================] - 0s 13ms/step - loss: 1.0730 - main_output_loss: 0.7282 - aux_output_loss: 0.6896 - main_output_acc: 0.3481 - aux_output_acc: 0.5481 - val_loss: 1.0670 - val_main_output_loss: 0.7215 - val_aux_output_loss: 0.6910 - val_main_output_acc: 0.3333 - val_aux_output_acc: 0.4667\n",
      "Epoch 4/300\n",
      "5/5 [==============================] - 0s 14ms/step - loss: 1.0627 - main_output_loss: 0.7246 - aux_output_loss: 0.6762 - main_output_acc: 0.3778 - aux_output_acc: 0.6148 - val_loss: 1.0576 - val_main_output_loss: 0.7165 - val_aux_output_loss: 0.6821 - val_main_output_acc: 0.3333 - val_aux_output_acc: 0.5333\n",
      "Epoch 5/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 1.0516 - main_output_loss: 0.7193 - aux_output_loss: 0.6647 - main_output_acc: 0.3778 - aux_output_acc: 0.6741 - val_loss: 1.0499 - val_main_output_loss: 0.7122 - val_aux_output_loss: 0.6753 - val_main_output_acc: 0.3333 - val_aux_output_acc: 0.6667\n",
      "Epoch 6/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 1.0363 - main_output_loss: 0.7077 - aux_output_loss: 0.6572 - main_output_acc: 0.4000 - aux_output_acc: 0.6593 - val_loss: 1.0428 - val_main_output_loss: 0.7080 - val_aux_output_loss: 0.6697 - val_main_output_acc: 0.3333 - val_aux_output_acc: 0.6667\n",
      "Epoch 7/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 1.0347 - main_output_loss: 0.7092 - aux_output_loss: 0.6509 - main_output_acc: 0.3704 - aux_output_acc: 0.6593 - val_loss: 1.0365 - val_main_output_loss: 0.7040 - val_aux_output_loss: 0.6650 - val_main_output_acc: 0.2667 - val_aux_output_acc: 0.6667\n",
      "Epoch 8/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 1.0239 - main_output_loss: 0.7029 - aux_output_loss: 0.6420 - main_output_acc: 0.4148 - aux_output_acc: 0.6963 - val_loss: 1.0308 - val_main_output_loss: 0.7003 - val_aux_output_loss: 0.6610 - val_main_output_acc: 0.4000 - val_aux_output_acc: 0.6667\n",
      "Epoch 9/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 1.0198 - main_output_loss: 0.6975 - aux_output_loss: 0.6446 - main_output_acc: 0.4815 - aux_output_acc: 0.6889 - val_loss: 1.0247 - val_main_output_loss: 0.6959 - val_aux_output_loss: 0.6576 - val_main_output_acc: 0.4667 - val_aux_output_acc: 0.6667\n",
      "Epoch 10/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 1.0173 - main_output_loss: 0.6946 - aux_output_loss: 0.6453 - main_output_acc: 0.5037 - aux_output_acc: 0.6741 - val_loss: 1.0197 - val_main_output_loss: 0.6925 - val_aux_output_loss: 0.6544 - val_main_output_acc: 0.6000 - val_aux_output_acc: 0.6667\n",
      "Epoch 11/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 1.0128 - main_output_loss: 0.6913 - aux_output_loss: 0.6430 - main_output_acc: 0.5704 - aux_output_acc: 0.6741 - val_loss: 1.0148 - val_main_output_loss: 0.6894 - val_aux_output_loss: 0.6509 - val_main_output_acc: 0.6000 - val_aux_output_acc: 0.6667\n",
      "Epoch 12/300\n",
      "5/5 [==============================] - 0s 20ms/step - loss: 1.0077 - main_output_loss: 0.6899 - aux_output_loss: 0.6357 - main_output_acc: 0.6222 - aux_output_acc: 0.6815 - val_loss: 1.0118 - val_main_output_loss: 0.6869 - val_aux_output_loss: 0.6497 - val_main_output_acc: 0.6000 - val_aux_output_acc: 0.6667\n",
      "Epoch 13/300\n",
      "5/5 [==============================] - 0s 21ms/step - loss: 0.9976 - main_output_loss: 0.6879 - aux_output_loss: 0.6194 - main_output_acc: 0.6222 - aux_output_acc: 0.6963 - val_loss: 1.0097 - val_main_output_loss: 0.6854 - val_aux_output_loss: 0.6486 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 14/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9980 - main_output_loss: 0.6865 - aux_output_loss: 0.6229 - main_output_acc: 0.6222 - aux_output_acc: 0.6963 - val_loss: 1.0085 - val_main_output_loss: 0.6840 - val_aux_output_loss: 0.6490 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 15/300\n",
      "5/5 [==============================] - 0s 19ms/step - loss: 0.9949 - main_output_loss: 0.6843 - aux_output_loss: 0.6212 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 1.0075 - val_main_output_loss: 0.6828 - val_aux_output_loss: 0.6494 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 16/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 0.9968 - main_output_loss: 0.6837 - aux_output_loss: 0.6263 - main_output_acc: 0.6667 - aux_output_acc: 0.6667 - val_loss: 1.0071 - val_main_output_loss: 0.6818 - val_aux_output_loss: 0.6505 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 17/300\n",
      "5/5 [==============================] - 0s 19ms/step - loss: 0.9899 - main_output_loss: 0.6820 - aux_output_loss: 0.6158 - main_output_acc: 0.6741 - aux_output_acc: 0.6815 - val_loss: 1.0078 - val_main_output_loss: 0.6812 - val_aux_output_loss: 0.6533 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 18/300\n",
      "5/5 [==============================] - 0s 20ms/step - loss: 0.9937 - main_output_loss: 0.6820 - aux_output_loss: 0.6234 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 1.0088 - val_main_output_loss: 0.6806 - val_aux_output_loss: 0.6565 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 19/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 0.9972 - main_output_loss: 0.6818 - aux_output_loss: 0.6310 - main_output_acc: 0.6593 - aux_output_acc: 0.6667 - val_loss: 1.0092 - val_main_output_loss: 0.6800 - val_aux_output_loss: 0.6585 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 20/300\n",
      "5/5 [==============================] - 0s 19ms/step - loss: 0.9890 - main_output_loss: 0.6817 - aux_output_loss: 0.6147 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 1.0087 - val_main_output_loss: 0.6795 - val_aux_output_loss: 0.6584 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 21/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 0.9894 - main_output_loss: 0.6798 - aux_output_loss: 0.6192 - main_output_acc: 0.6741 - aux_output_acc: 0.6815 - val_loss: 1.0091 - val_main_output_loss: 0.6789 - val_aux_output_loss: 0.6604 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 22/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9891 - main_output_loss: 0.6802 - aux_output_loss: 0.6178 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 1.0089 - val_main_output_loss: 0.6784 - val_aux_output_loss: 0.6609 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 23/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9927 - main_output_loss: 0.6795 - aux_output_loss: 0.6265 - main_output_acc: 0.6667 - aux_output_acc: 0.6667 - val_loss: 1.0089 - val_main_output_loss: 0.6779 - val_aux_output_loss: 0.6620 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 24/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 0.9868 - main_output_loss: 0.6787 - aux_output_loss: 0.6162 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 1.0078 - val_main_output_loss: 0.6774 - val_aux_output_loss: 0.6608 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 25/300\n",
      "5/5 [==============================] - 0s 19ms/step - loss: 0.9918 - main_output_loss: 0.6778 - aux_output_loss: 0.6280 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 1.0074 - val_main_output_loss: 0.6769 - val_aux_output_loss: 0.6610 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 26/300\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9796 - main_output_loss: 0.6782 - aux_output_loss: 0.6030 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 1.0078 - val_main_output_loss: 0.6764 - val_aux_output_loss: 0.6627 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 27/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9792 - main_output_loss: 0.6767 - aux_output_loss: 0.6051 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 1.0069 - val_main_output_loss: 0.6760 - val_aux_output_loss: 0.6618 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 28/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9904 - main_output_loss: 0.6766 - aux_output_loss: 0.6276 - main_output_acc: 0.6741 - aux_output_acc: 0.6593 - val_loss: 1.0052 - val_main_output_loss: 0.6756 - val_aux_output_loss: 0.6591 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 29/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9909 - main_output_loss: 0.6760 - aux_output_loss: 0.6298 - main_output_acc: 0.6667 - aux_output_acc: 0.6667 - val_loss: 1.0041 - val_main_output_loss: 0.6752 - val_aux_output_loss: 0.6578 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 30/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9833 - main_output_loss: 0.6758 - aux_output_loss: 0.6150 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 1.0031 - val_main_output_loss: 0.6748 - val_aux_output_loss: 0.6566 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 31/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9741 - main_output_loss: 0.6745 - aux_output_loss: 0.5991 - main_output_acc: 0.6741 - aux_output_acc: 0.6889 - val_loss: 1.0017 - val_main_output_loss: 0.6746 - val_aux_output_loss: 0.6541 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 32/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9803 - main_output_loss: 0.6739 - aux_output_loss: 0.6128 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9999 - val_main_output_loss: 0.6744 - val_aux_output_loss: 0.6510 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 33/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9773 - main_output_loss: 0.6738 - aux_output_loss: 0.6069 - main_output_acc: 0.6741 - aux_output_acc: 0.6815 - val_loss: 0.9992 - val_main_output_loss: 0.6739 - val_aux_output_loss: 0.6507 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 34/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9768 - main_output_loss: 0.6739 - aux_output_loss: 0.6058 - main_output_acc: 0.6667 - aux_output_acc: 0.6889 - val_loss: 0.9985 - val_main_output_loss: 0.6734 - val_aux_output_loss: 0.6502 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 35/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9783 - main_output_loss: 0.6725 - aux_output_loss: 0.6116 - main_output_acc: 0.6741 - aux_output_acc: 0.6593 - val_loss: 0.9975 - val_main_output_loss: 0.6728 - val_aux_output_loss: 0.6494 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 36/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9751 - main_output_loss: 0.6729 - aux_output_loss: 0.6044 - main_output_acc: 0.6667 - aux_output_acc: 0.6889 - val_loss: 0.9968 - val_main_output_loss: 0.6724 - val_aux_output_loss: 0.6488 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 37/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9781 - main_output_loss: 0.6727 - aux_output_loss: 0.6107 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 0.9964 - val_main_output_loss: 0.6720 - val_aux_output_loss: 0.6489 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 38/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9754 - main_output_loss: 0.6714 - aux_output_loss: 0.6080 - main_output_acc: 0.6667 - aux_output_acc: 0.6667 - val_loss: 0.9962 - val_main_output_loss: 0.6716 - val_aux_output_loss: 0.6491 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 39/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9770 - main_output_loss: 0.6718 - aux_output_loss: 0.6105 - main_output_acc: 0.6667 - aux_output_acc: 0.6889 - val_loss: 0.9957 - val_main_output_loss: 0.6714 - val_aux_output_loss: 0.6485 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 40/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9771 - main_output_loss: 0.6709 - aux_output_loss: 0.6123 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9953 - val_main_output_loss: 0.6711 - val_aux_output_loss: 0.6484 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 41/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 0.9728 - main_output_loss: 0.6700 - aux_output_loss: 0.6055 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9953 - val_main_output_loss: 0.6707 - val_aux_output_loss: 0.6492 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 42/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9814 - main_output_loss: 0.6699 - aux_output_loss: 0.6229 - main_output_acc: 0.6741 - aux_output_acc: 0.6741 - val_loss: 0.9954 - val_main_output_loss: 0.6702 - val_aux_output_loss: 0.6503 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 43/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9753 - main_output_loss: 0.6683 - aux_output_loss: 0.6139 - main_output_acc: 0.6741 - aux_output_acc: 0.7037 - val_loss: 0.9948 - val_main_output_loss: 0.6698 - val_aux_output_loss: 0.6500 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 44/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9779 - main_output_loss: 0.6699 - aux_output_loss: 0.6160 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 0.9945 - val_main_output_loss: 0.6692 - val_aux_output_loss: 0.6506 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 45/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9741 - main_output_loss: 0.6688 - aux_output_loss: 0.6105 - main_output_acc: 0.6667 - aux_output_acc: 0.6593 - val_loss: 0.9938 - val_main_output_loss: 0.6688 - val_aux_output_loss: 0.6501 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 46/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9714 - main_output_loss: 0.6681 - aux_output_loss: 0.6066 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9937 - val_main_output_loss: 0.6684 - val_aux_output_loss: 0.6506 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 47/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9707 - main_output_loss: 0.6670 - aux_output_loss: 0.6073 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 0.9928 - val_main_output_loss: 0.6683 - val_aux_output_loss: 0.6490 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 48/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9745 - main_output_loss: 0.6669 - aux_output_loss: 0.6154 - main_output_acc: 0.6741 - aux_output_acc: 0.6667 - val_loss: 0.9914 - val_main_output_loss: 0.6681 - val_aux_output_loss: 0.6467 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 49/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9599 - main_output_loss: 0.6660 - aux_output_loss: 0.5879 - main_output_acc: 0.6741 - aux_output_acc: 0.6889 - val_loss: 0.9907 - val_main_output_loss: 0.6678 - val_aux_output_loss: 0.6458 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 50/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9747 - main_output_loss: 0.6658 - aux_output_loss: 0.6177 - main_output_acc: 0.6741 - aux_output_acc: 0.6667 - val_loss: 0.9905 - val_main_output_loss: 0.6672 - val_aux_output_loss: 0.6466 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 51/300\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9635 - main_output_loss: 0.6655 - aux_output_loss: 0.5961 - main_output_acc: 0.6741 - aux_output_acc: 0.6963 - val_loss: 0.9901 - val_main_output_loss: 0.6666 - val_aux_output_loss: 0.6471 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 52/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9701 - main_output_loss: 0.6654 - aux_output_loss: 0.6095 - main_output_acc: 0.6667 - aux_output_acc: 0.6889 - val_loss: 0.9897 - val_main_output_loss: 0.6663 - val_aux_output_loss: 0.6469 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 53/300\n",
      "5/5 [==============================] - 0s 19ms/step - loss: 0.9658 - main_output_loss: 0.6648 - aux_output_loss: 0.6019 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 0.9895 - val_main_output_loss: 0.6658 - val_aux_output_loss: 0.6474 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 54/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 0.9627 - main_output_loss: 0.6651 - aux_output_loss: 0.5952 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9885 - val_main_output_loss: 0.6653 - val_aux_output_loss: 0.6464 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 55/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9606 - main_output_loss: 0.6627 - aux_output_loss: 0.5957 - main_output_acc: 0.6741 - aux_output_acc: 0.6963 - val_loss: 0.9873 - val_main_output_loss: 0.6650 - val_aux_output_loss: 0.6446 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 56/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9675 - main_output_loss: 0.6628 - aux_output_loss: 0.6094 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9862 - val_main_output_loss: 0.6648 - val_aux_output_loss: 0.6428 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 57/300\n",
      "5/5 [==============================] - 0s 20ms/step - loss: 0.9648 - main_output_loss: 0.6625 - aux_output_loss: 0.6045 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 0.9851 - val_main_output_loss: 0.6647 - val_aux_output_loss: 0.6407 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 58/300\n",
      "5/5 [==============================] - 0s 18ms/step - loss: 0.9666 - main_output_loss: 0.6630 - aux_output_loss: 0.6072 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9839 - val_main_output_loss: 0.6650 - val_aux_output_loss: 0.6380 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 59/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9618 - main_output_loss: 0.6618 - aux_output_loss: 0.6000 - main_output_acc: 0.6667 - aux_output_acc: 0.7185 - val_loss: 0.9832 - val_main_output_loss: 0.6651 - val_aux_output_loss: 0.6362 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 60/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9669 - main_output_loss: 0.6623 - aux_output_loss: 0.6090 - main_output_acc: 0.6667 - aux_output_acc: 0.6667 - val_loss: 0.9831 - val_main_output_loss: 0.6650 - val_aux_output_loss: 0.6362 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 61/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9597 - main_output_loss: 0.6623 - aux_output_loss: 0.5947 - main_output_acc: 0.6667 - aux_output_acc: 0.6889 - val_loss: 0.9823 - val_main_output_loss: 0.6651 - val_aux_output_loss: 0.6343 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 62/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9669 - main_output_loss: 0.6612 - aux_output_loss: 0.6113 - main_output_acc: 0.6741 - aux_output_acc: 0.6741 - val_loss: 0.9813 - val_main_output_loss: 0.6652 - val_aux_output_loss: 0.6321 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 63/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9627 - main_output_loss: 0.6611 - aux_output_loss: 0.6032 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 0.9801 - val_main_output_loss: 0.6650 - val_aux_output_loss: 0.6302 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 64/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9580 - main_output_loss: 0.6607 - aux_output_loss: 0.5947 - main_output_acc: 0.6667 - aux_output_acc: 0.6593 - val_loss: 0.9797 - val_main_output_loss: 0.6647 - val_aux_output_loss: 0.6299 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 65/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9540 - main_output_loss: 0.6599 - aux_output_loss: 0.5882 - main_output_acc: 0.6741 - aux_output_acc: 0.7037 - val_loss: 0.9786 - val_main_output_loss: 0.6645 - val_aux_output_loss: 0.6281 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 66/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9559 - main_output_loss: 0.6607 - aux_output_loss: 0.5905 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9786 - val_main_output_loss: 0.6642 - val_aux_output_loss: 0.6289 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 67/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9591 - main_output_loss: 0.6605 - aux_output_loss: 0.5973 - main_output_acc: 0.6667 - aux_output_acc: 0.7111 - val_loss: 0.9778 - val_main_output_loss: 0.6642 - val_aux_output_loss: 0.6273 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 68/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9498 - main_output_loss: 0.6587 - aux_output_loss: 0.5823 - main_output_acc: 0.6741 - aux_output_acc: 0.7037 - val_loss: 0.9773 - val_main_output_loss: 0.6637 - val_aux_output_loss: 0.6272 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 69/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9530 - main_output_loss: 0.6592 - aux_output_loss: 0.5875 - main_output_acc: 0.6667 - aux_output_acc: 0.7111 - val_loss: 0.9772 - val_main_output_loss: 0.6632 - val_aux_output_loss: 0.6280 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 70/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9529 - main_output_loss: 0.6595 - aux_output_loss: 0.5868 - main_output_acc: 0.6667 - aux_output_acc: 0.7111 - val_loss: 0.9769 - val_main_output_loss: 0.6627 - val_aux_output_loss: 0.6283 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 71/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9547 - main_output_loss: 0.6571 - aux_output_loss: 0.5951 - main_output_acc: 0.6741 - aux_output_acc: 0.7111 - val_loss: 0.9766 - val_main_output_loss: 0.6625 - val_aux_output_loss: 0.6283 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 72/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9517 - main_output_loss: 0.6566 - aux_output_loss: 0.5902 - main_output_acc: 0.6741 - aux_output_acc: 0.6815 - val_loss: 0.9765 - val_main_output_loss: 0.6625 - val_aux_output_loss: 0.6279 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 73/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9564 - main_output_loss: 0.6564 - aux_output_loss: 0.6001 - main_output_acc: 0.6741 - aux_output_acc: 0.6667 - val_loss: 0.9765 - val_main_output_loss: 0.6624 - val_aux_output_loss: 0.6281 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 74/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9544 - main_output_loss: 0.6565 - aux_output_loss: 0.5958 - main_output_acc: 0.6741 - aux_output_acc: 0.7037 - val_loss: 0.9759 - val_main_output_loss: 0.6620 - val_aux_output_loss: 0.6278 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 75/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9584 - main_output_loss: 0.6568 - aux_output_loss: 0.6032 - main_output_acc: 0.6741 - aux_output_acc: 0.6667 - val_loss: 0.9745 - val_main_output_loss: 0.6614 - val_aux_output_loss: 0.6263 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 76/300\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9474 - main_output_loss: 0.6569 - aux_output_loss: 0.5810 - main_output_acc: 0.6667 - aux_output_acc: 0.6667 - val_loss: 0.9743 - val_main_output_loss: 0.6600 - val_aux_output_loss: 0.6287 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 77/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9531 - main_output_loss: 0.6571 - aux_output_loss: 0.5920 - main_output_acc: 0.6667 - aux_output_acc: 0.7185 - val_loss: 0.9744 - val_main_output_loss: 0.6592 - val_aux_output_loss: 0.6304 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 78/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9511 - main_output_loss: 0.6549 - aux_output_loss: 0.5925 - main_output_acc: 0.6741 - aux_output_acc: 0.6889 - val_loss: 0.9730 - val_main_output_loss: 0.6592 - val_aux_output_loss: 0.6276 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 79/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9510 - main_output_loss: 0.6538 - aux_output_loss: 0.5943 - main_output_acc: 0.6741 - aux_output_acc: 0.7185 - val_loss: 0.9725 - val_main_output_loss: 0.6590 - val_aux_output_loss: 0.6269 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 80/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9503 - main_output_loss: 0.6560 - aux_output_loss: 0.5886 - main_output_acc: 0.6667 - aux_output_acc: 0.7111 - val_loss: 0.9721 - val_main_output_loss: 0.6588 - val_aux_output_loss: 0.6265 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 81/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9435 - main_output_loss: 0.6547 - aux_output_loss: 0.5775 - main_output_acc: 0.6741 - aux_output_acc: 0.7185 - val_loss: 0.9707 - val_main_output_loss: 0.6595 - val_aux_output_loss: 0.6224 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 82/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9543 - main_output_loss: 0.6536 - aux_output_loss: 0.6014 - main_output_acc: 0.6667 - aux_output_acc: 0.6889 - val_loss: 0.9704 - val_main_output_loss: 0.6598 - val_aux_output_loss: 0.6212 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 83/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9541 - main_output_loss: 0.6556 - aux_output_loss: 0.5971 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 0.9702 - val_main_output_loss: 0.6601 - val_aux_output_loss: 0.6203 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 84/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9442 - main_output_loss: 0.6525 - aux_output_loss: 0.5833 - main_output_acc: 0.6741 - aux_output_acc: 0.7037 - val_loss: 0.9707 - val_main_output_loss: 0.6596 - val_aux_output_loss: 0.6222 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 85/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9372 - main_output_loss: 0.6552 - aux_output_loss: 0.5641 - main_output_acc: 0.6667 - aux_output_acc: 0.7037 - val_loss: 0.9709 - val_main_output_loss: 0.6592 - val_aux_output_loss: 0.6234 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 86/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9598 - main_output_loss: 0.6554 - aux_output_loss: 0.6089 - main_output_acc: 0.6667 - aux_output_acc: 0.7037 - val_loss: 0.9716 - val_main_output_loss: 0.6587 - val_aux_output_loss: 0.6257 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 87/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9569 - main_output_loss: 0.6522 - aux_output_loss: 0.6094 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 0.9724 - val_main_output_loss: 0.6585 - val_aux_output_loss: 0.6279 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 88/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9615 - main_output_loss: 0.6560 - aux_output_loss: 0.6110 - main_output_acc: 0.6667 - aux_output_acc: 0.6667 - val_loss: 0.9723 - val_main_output_loss: 0.6593 - val_aux_output_loss: 0.6259 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 89/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9463 - main_output_loss: 0.6534 - aux_output_loss: 0.5857 - main_output_acc: 0.6741 - aux_output_acc: 0.6889 - val_loss: 0.9714 - val_main_output_loss: 0.6599 - val_aux_output_loss: 0.6231 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 90/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9391 - main_output_loss: 0.6513 - aux_output_loss: 0.5756 - main_output_acc: 0.6741 - aux_output_acc: 0.7111 - val_loss: 0.9711 - val_main_output_loss: 0.6599 - val_aux_output_loss: 0.6223 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 91/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9567 - main_output_loss: 0.6529 - aux_output_loss: 0.6077 - main_output_acc: 0.6667 - aux_output_acc: 0.6889 - val_loss: 0.9702 - val_main_output_loss: 0.6597 - val_aux_output_loss: 0.6210 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 92/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9379 - main_output_loss: 0.6501 - aux_output_loss: 0.5756 - main_output_acc: 0.6667 - aux_output_acc: 0.7185 - val_loss: 0.9693 - val_main_output_loss: 0.6597 - val_aux_output_loss: 0.6193 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 93/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9419 - main_output_loss: 0.6529 - aux_output_loss: 0.5780 - main_output_acc: 0.6667 - aux_output_acc: 0.7185 - val_loss: 0.9695 - val_main_output_loss: 0.6595 - val_aux_output_loss: 0.6200 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 94/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9363 - main_output_loss: 0.6510 - aux_output_loss: 0.5705 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 0.9701 - val_main_output_loss: 0.6592 - val_aux_output_loss: 0.6218 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 95/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9477 - main_output_loss: 0.6521 - aux_output_loss: 0.5912 - main_output_acc: 0.6667 - aux_output_acc: 0.6741 - val_loss: 0.9705 - val_main_output_loss: 0.6588 - val_aux_output_loss: 0.6233 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 96/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9441 - main_output_loss: 0.6517 - aux_output_loss: 0.5849 - main_output_acc: 0.6667 - aux_output_acc: 0.6963 - val_loss: 0.9716 - val_main_output_loss: 0.6588 - val_aux_output_loss: 0.6257 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 97/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9533 - main_output_loss: 0.6501 - aux_output_loss: 0.6062 - main_output_acc: 0.6741 - aux_output_acc: 0.6815 - val_loss: 0.9716 - val_main_output_loss: 0.6589 - val_aux_output_loss: 0.6255 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 98/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9464 - main_output_loss: 0.6469 - aux_output_loss: 0.5989 - main_output_acc: 0.6741 - aux_output_acc: 0.6593 - val_loss: 0.9717 - val_main_output_loss: 0.6583 - val_aux_output_loss: 0.6268 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 99/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9411 - main_output_loss: 0.6494 - aux_output_loss: 0.5834 - main_output_acc: 0.6741 - aux_output_acc: 0.7037 - val_loss: 0.9714 - val_main_output_loss: 0.6582 - val_aux_output_loss: 0.6264 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 100/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9360 - main_output_loss: 0.6490 - aux_output_loss: 0.5739 - main_output_acc: 0.6741 - aux_output_acc: 0.6815 - val_loss: 0.9701 - val_main_output_loss: 0.6586 - val_aux_output_loss: 0.6229 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 101/300\n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9391 - main_output_loss: 0.6489 - aux_output_loss: 0.5805 - main_output_acc: 0.6741 - aux_output_acc: 0.6889 - val_loss: 0.9695 - val_main_output_loss: 0.6587 - val_aux_output_loss: 0.6217 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 102/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9445 - main_output_loss: 0.6517 - aux_output_loss: 0.5855 - main_output_acc: 0.6593 - aux_output_acc: 0.6815 - val_loss: 0.9685 - val_main_output_loss: 0.6587 - val_aux_output_loss: 0.6196 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 103/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9355 - main_output_loss: 0.6472 - aux_output_loss: 0.5767 - main_output_acc: 0.6741 - aux_output_acc: 0.6963 - val_loss: 0.9681 - val_main_output_loss: 0.6586 - val_aux_output_loss: 0.6190 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 104/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9405 - main_output_loss: 0.6505 - aux_output_loss: 0.5800 - main_output_acc: 0.6593 - aux_output_acc: 0.6889 - val_loss: 0.9686 - val_main_output_loss: 0.6582 - val_aux_output_loss: 0.6207 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 105/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9432 - main_output_loss: 0.6485 - aux_output_loss: 0.5893 - main_output_acc: 0.6741 - aux_output_acc: 0.6963 - val_loss: 0.9684 - val_main_output_loss: 0.6578 - val_aux_output_loss: 0.6211 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 106/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9337 - main_output_loss: 0.6480 - aux_output_loss: 0.5715 - main_output_acc: 0.6741 - aux_output_acc: 0.7037 - val_loss: 0.9683 - val_main_output_loss: 0.6572 - val_aux_output_loss: 0.6221 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 107/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9276 - main_output_loss: 0.6480 - aux_output_loss: 0.5592 - main_output_acc: 0.6741 - aux_output_acc: 0.7185 - val_loss: 0.9665 - val_main_output_loss: 0.6569 - val_aux_output_loss: 0.6193 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 108/300\n",
      "5/5 [==============================] - 0s 16ms/step - loss: 0.9417 - main_output_loss: 0.6496 - aux_output_loss: 0.5842 - main_output_acc: 0.6667 - aux_output_acc: 0.6815 - val_loss: 0.9640 - val_main_output_loss: 0.6569 - val_aux_output_loss: 0.6142 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 109/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9403 - main_output_loss: 0.6465 - aux_output_loss: 0.5876 - main_output_acc: 0.6741 - aux_output_acc: 0.6963 - val_loss: 0.9623 - val_main_output_loss: 0.6572 - val_aux_output_loss: 0.6101 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 110/300\n",
      "5/5 [==============================] - 0s 17ms/step - loss: 0.9432 - main_output_loss: 0.6450 - aux_output_loss: 0.5964 - main_output_acc: 0.6741 - aux_output_acc: 0.7185 - val_loss: 0.9607 - val_main_output_loss: 0.6572 - val_aux_output_loss: 0.6070 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 111/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9344 - main_output_loss: 0.6455 - aux_output_loss: 0.5777 - main_output_acc: 0.6741 - aux_output_acc: 0.6889 - val_loss: 0.9599 - val_main_output_loss: 0.6570 - val_aux_output_loss: 0.6058 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 112/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9404 - main_output_loss: 0.6473 - aux_output_loss: 0.5861 - main_output_acc: 0.6815 - aux_output_acc: 0.6741 - val_loss: 0.9593 - val_main_output_loss: 0.6569 - val_aux_output_loss: 0.6048 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 113/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9250 - main_output_loss: 0.6446 - aux_output_loss: 0.5607 - main_output_acc: 0.6741 - aux_output_acc: 0.7111 - val_loss: 0.9582 - val_main_output_loss: 0.6567 - val_aux_output_loss: 0.6030 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 114/300\n",
      "5/5 [==============================] - 0s 14ms/step - loss: 0.9422 - main_output_loss: 0.6447 - aux_output_loss: 0.5949 - main_output_acc: 0.6815 - aux_output_acc: 0.6741 - val_loss: 0.9565 - val_main_output_loss: 0.6563 - val_aux_output_loss: 0.6005 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 115/300\n",
      "5/5 [==============================] - 0s 14ms/step - loss: 0.9219 - main_output_loss: 0.6436 - aux_output_loss: 0.5566 - main_output_acc: 0.6815 - aux_output_acc: 0.7037 - val_loss: 0.9554 - val_main_output_loss: 0.6557 - val_aux_output_loss: 0.5995 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 116/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9372 - main_output_loss: 0.6464 - aux_output_loss: 0.5814 - main_output_acc: 0.6815 - aux_output_acc: 0.7185 - val_loss: 0.9556 - val_main_output_loss: 0.6549 - val_aux_output_loss: 0.6013 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 117/300\n",
      "5/5 [==============================] - 0s 15ms/step - loss: 0.9226 - main_output_loss: 0.6420 - aux_output_loss: 0.5612 - main_output_acc: 0.6815 - aux_output_acc: 0.7259 - val_loss: 0.9538 - val_main_output_loss: 0.6543 - val_aux_output_loss: 0.5989 - val_main_output_acc: 0.6667 - val_aux_output_acc: 0.6667\n",
      "Epoch 118/300\n",
      "1/5 [=====>........................] - ETA: 0s - loss: 0.9260 - main_output_loss: 0.6521 - aux_output_loss: 0.5477 - main_output_acc: 0.6562 - aux_output_acc: 0.6562"
     ]
    }
   ],
   "source": [
    "import keras.backend as K\n",
    "# from keras.callbacks import ReduceLROnPlateau,ModelCheckpoint\n",
    "# from keras.optimizers import RMSprop\n",
    "\n",
    "\n",
    "# reduce_lr = ReduceLROnPlateau(monitor='val_main_output_loss', factor=0.9, patience=30, min_lr=0.000001, verbose=1)\n",
    "# checkpointer = ModelCheckpoint(filepath='lstm.hdf5', verbose=1, save_best_only=True)\n",
    "\n",
    "history=model.fit({'main_input': dframe_train_X_time.astype(np.float32), 'aux_input': dframe_train_static.astype(np.float32)},\n",
    "          {'main_output':dframe_train_Y.astype(np.float32), 'aux_output': dframe_train_Y.astype(np.float32)},\n",
    "          epochs=300,validation_split=0.10)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "5a7cf7fe",
   "metadata": {},
   "outputs": [],
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
   "execution_count": null,
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
   "execution_count": null,
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
   "execution_count": null,
   "id": "2e041faa",
   "metadata": {},
   "outputs": [],
   "source": [
    "# main auc\n",
    "compute_roc(dframe_test_Y,y_score[0],'deep learning')"
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
