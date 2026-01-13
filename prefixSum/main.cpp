#include <iostream>
#include <vector>

int prefixSumCpu(std::vector<int>& arr){
    arr.insert(arr.begin(), 0);
    for(int i = 1; i<arr.size(); i++){
        arr[i] = arr[i]+arr[i-1];
    }
    return arr[arr.size()-1];
}
int main() {
    std::vector<int> arr = {1,2,3,4,5,6,7,8,9,10};
    int result = prefixSumCpu(arr);
    std::cout << result<<std::endl;
    return 0;
}