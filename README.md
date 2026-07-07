This project utilizes machine learning to predict optimal lipid nanoparticle (LNP) components ratios and pH for circular mRNA delivery. The data is an experimental data from our lab. The Ionizable lipid and PEG lipid are novelly synthesized by our college. Therefore, we want to evaluate how these two components affect the LNP properties and optimize the LNP formula.

## 1. Data exploration  
Firstly, the data was inspected for its shape, type, some statistics and missing value.  
```
**********************************************************************
Shape of dataset :  (25, 13)
**********************************************************************
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 25 entries, 0 to 24
Data columns (total 13 columns):
 #   Column               Non-Null Count  Dtype
---  ------               --------------  -----
 0   Run                  25 non-null     object
 1   PEG lipid (%)        25 non-null     float64
 2   Helper lipid (%)     25 non-null     float64
 3   Cholesterol (%)      25 non-null     float64
 4   Ionizable lipid (%)  25 non-null     float64
 5   pH                   25 non-null     int64
 6   Size (nm)            25 non-null     float64
 7   PDI                  25 non-null     float64
 8   %EE                  25 non-null     float64
 9   MFI                  25 non-null     float64
 10  %Positive cells      25 non-null     float64
 11  non-lysed LNP        25 non-null     float64
 12  Lysed LNP            25 non-null     float64
dtypes: float64(11), int64(1), object(1)
memory usage: 2.7+ KB
None
**********************************************************************
STATISTICAL ANALYSIS OF NUMERICAL DATA
**********************************************************************
                     count          mean          std           min          25%           50%           75%           max
PEG lipid (%)         25.0      1.500000     0.353553      1.000000      1.50000      1.500000      1.500000      2.000000
Helper lipid (%)      25.0     15.000000     1.767767     12.500000     15.00000     15.000000     15.000000     17.500000
Cholesterol (%)       25.0     38.500000     3.535534     33.500000     38.50000     38.500000     38.500000     43.500000
Ionizable lipid (%)   25.0     45.000000     3.968627     37.500000     42.50000     45.000000     47.500000     52.500000
pH                    25.0      4.000000     0.707107      3.000000      4.00000      4.000000      4.000000      5.000000
Size (nm)             25.0    153.936000    13.940644    113.400000    148.10000    155.500000    158.800000    193.000000
PDI                   25.0      0.102744     0.040533      0.008757      0.07375      0.112000      0.127300      0.164600
%EE                   25.0     73.107526     4.704535     64.840869     69.85434     73.371222     76.921203     83.080464
MFI                   25.0  31521.253200  9796.212490  17653.000000  24884.00000  29764.000000  36828.000000  50779.000000
%Positive cells       25.0     62.070800    11.829510     41.200000     56.90000     63.300000     68.300000     81.800000
non-lysed LNP         25.0     27.269040     4.990211     18.378000     23.50900     27.206000     30.623000     37.372000
Lysed LNP             25.0    101.947600    11.483685     75.880000     95.03000     99.930000    111.180000    118.770000
**********************************************************************
STATISTICAL ANALYSIS OF CATEGORICAL DATA
**********************************************************************
    count unique top freq
Run    25     25   A    1
**********************************************************************
MISSING VALUES
**********************************************************************
Run                    0
PEG lipid (%)          0
Helper lipid (%)       0
Cholesterol (%)        0
Ionizable lipid (%)    0
pH                     0
Size (nm)              0
PDI                    0
%EE                    0
MFI                    0
%Positive cells        0
non-lysed LNP          0
Lysed LNP              0
dtype: int64
**********************************************************************
MISSING VALUES IN %
**********************************************************************
Run                    0.0
PEG lipid (%)          0.0
Helper lipid (%)       0.0
Cholesterol (%)        0.0
Ionizable lipid (%)    0.0
pH                     0.0
Size (nm)              0.0
PDI                    0.0
%EE                    0.0
MFI                    0.0
%Positive cells        0.0
non-lysed LNP          0.0
Lysed LNP              0.0
dtype: float64
**********************************************************************`
```  

## 2. Data visualization  
After the data was inspected, I used the scatter plot to visualize the data pattern of each inputs and outputs.

<p align="center">
<img width="1200" height="500" alt="MFI" src="https://github.com/user-attachments/assets/2285da7b-9078-4dbc-bb81-c592010ad5ed" />  
</p>

From the plot between inputs and mean fluorescence intensity (MFI), which is the main output that we need to maximize, it is shown that all inputs except cholesterol affect the MFI.  

## 3. Correlation heatmap
The correlation heatmap of input parameters aginst output parameters is generated. The asterisks represent p value where * is p <= 0.05, ** is p <= 0.01 and *** is p <= 0.001.  

<p align="center">
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/6551df60-f485-4055-ba3e-0d54ed2dd0e6" />  
</p>

The heatmap displays the same trend as the scatter plot. the pH and Helper lipid shows significant positive correlation with MFI, while Ionizable lipid and PEG lipid shows negative correlation.  

## 4. Model selection  
In order to select the best model for predicting MFI, nested cross validation, together with hyperparameter optimization, was performed. Leave one out cross validation was used as a outer cv method, since the dataset was very small, while RepeatedKFold was used as a inner cv method. Cholesterol was shown to have only a little correlation with the MFI, so I decided to withdraw it from the inputs. The model with the lowest root mean square error (RMSE) was selected.

<p align="center">
<img width="500" height="300" alt="image" src="https://github.com/user-attachments/assets/32c47ec7-4362-44ea-bd8f-6d3f88496ddf" />  
</p>

MLP model shows the lowest RMSE value. However, the prediction result from MLP model returns the same values for every formula which is unusual. It is due to the nature of the model that is not suitable with the small dataset. Therefore, the Elastic Net model, the second lowest RMSE, is selected instead.  

## 5. Learning curve plot  
After the model was selected, learning curve was plot to determine the model's performance as the training set size increases. LeavOneOut was used since the dataset is very small.

<p align="center">
<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/0923f3e7-f03a-442c-b8de-40e439bad1fb" />
</p>

As the training set size increses, the test score also increses and the dataset is very small, so I decided to split 15% of the data to be validation/test set.

## 6. Prediction  
The dataframe of LNP formulas was generated for every combinations of each input. Then, the prediction was carried out. In addition to MFI, the model also predict other outputs, including %Positive cells, %Encapsulation efficiancy, size in nm and PDI. I filtered the formula with lower MFI compared to the formula Q from the dataset (highest MFI) plus RMSE value from the test set out and shows only the top 5 predicted formulas with the highest MFI.

<p align="center">
<img width="1700" height="500" alt="image" src="https://github.com/user-attachments/assets/16cf5718-c35a-4ff1-88b7-569ae2faea21" />
</p>

## 7. Parity and residual plots
The parity and residual plot represented the performance of regression fit. The training (85%) and test (15%) set were separately evaluated.

<p align="center">
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/7538926f-8b55-4a34-a510-36216310c290" />
 <br>
<img width="500" height="400" alt="image" src="https://github.com/user-attachments/assets/2262b2db-061a-4adf-b179-fc96ca8b98ec" />
 <br>
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/cddec222-47d4-44c4-85ec-d659b1d748bd" /> 
 <br>
<img width="500" height="350" alt="image" src="https://github.com/user-attachments/assets/de4165ef-34e9-4772-8059-d923528cdfe3" />
</p>

## 8. Parameter weight
The weight of each parameter was also plot and shown below.

<p align="center">
<img width="1800" height="751" alt="image" src="https://github.com/user-attachments/assets/126be5fb-a01d-415e-b81e-41680d840f14" />
<br>
<br>
--------------------------------------------------------------------------------------------------------------------------------------------------
<br>
<br>
<p align="left">
**In conclusion, due to the small size of the data set, we cannot observe the turning point of MFI in each input. We can actually predict the optimized formula from just visuallized data. However, this workflow can be fundamentally used with other, larger datasets.**
</p>



