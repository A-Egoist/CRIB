# Dataset

We used [KuaiRand](https://kuairand.com/), [KuaiRec](https://kuairec.com/), and [KuaiSAR](https://kuaisar.github.io/). You can download these datasets and unzip them into current directory.

## Dataset Information

| Dataset       | #Users | #Items    | #Interactions | Date                    | Density |
| ------------- | ------ | --------- | ------------- | ----------------------- | ------- |
| KuaiRand-Pure | 27,285 | 7,551     | 1,436,609     | 2022.04.08 ~ 2022.05.08 | 0.6973% |
| KuaiRand-1K   | 1,000  | 4,369,953 | 11,713,045    | 2022.04.08 ~ 2022.05.08 | 0.2680% |
| KuaiRec-small | 1,411  | 3,327     | 4,676,570     | 2020.07.07 ~ 2020.09.05 | 99.6%   |
| KuaiRec-big   | 7,176  | 10,728    | 12,530,806    | 2020.07.07 ~ 2020.09.05 | 16.3%   |
| KuaiSAR-small | 25,877 | 2,281,034 | 7,493,101     | 2023.05.22 ~ 2023.06.10 | 0.0127% |
| KuaiSAR       | 25,877 | 4,046,367 | 14,605,716    | 2023.05.22 ~ 2023.06.10 | 0.0139% |



## KuaiRand

*   KuaiRand-Pure: Split the dataset based on interaction dates as follows:
    *   Training Set: 2022.04.08 ~ 2022.04.28 (20 days)
    *   Validation Set: 2022.04.29 ~ 2022.05.01 (3 days)
    *   Test Set: 2022.05.02 ~ 2022.05.05 (7 days)

*   KuaiRand-1K: Split the dataset based on interaction dates as follows:
    *   Training Set: 2022.04.08 ~ 2022.04.28 (20 days)
    *   Validation Set: 2022.04.29 ~ 2022.05.01 (3 days)
    *   Test Set: 2022.05.02 ~ 2022.05.05 (7 days)



## KuaiRec

*   KuaiRec-small
    *   Training Set: 2020.07.07 ~ 2020.08.14 (38 days)
    *   Validation Set: 2020.08.15 ~ 2020.08.21 (7 days)
    *   Test Set: 2020.08.22 ~ 2020.09.05 (14 days)

*   KuaiRec-big
    *   Training Set: 2020.07.07 ~ 2020.08.14 (38 days)
    *   Validation Set: 2020.08.15 ~ 2020.08.21 (7 days)
    *   Test Set: 2020.08.22 ~ 2020.09.05 (14 days)



## KuaiSAR

*   KuaiSAR-small
    *   Training Set: 2023.05.22 ~ 2023.05.27 (6 days)
    *   Validation Set: 2023.05.28 ~ 2023.05.28 (1 day)
    *   Test Set: 2023.05.29 ~ 2023.05.31 (3 days)

*   KuaiSAR
    *   Training Set: 2023.05.22 ~ 2023.06.02 (12 days)
    *   Validation Set: 2023.06.03 ~ 2023.06.04 (2 days)
    *   Test Set: 2023.06.05 ~ 2023.06.10 (6 days)

