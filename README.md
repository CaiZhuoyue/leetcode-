# leetcode刷题（持续更新）

整合了leetcode字节文档中的题目







#### 45.跳跃游戏2（独立完成，y总的思路无法理解）

我的方法：

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        vector<int> cnt(nums.size()+1,nums.size()+10); // 记录跳到i的时候需要的最小步数
        cnt[0]=0;

        for(int i=0;i<nums.size();i++){
            for(int j=1;j<=nums[i];j++){ // 遍历从i开始可以跳的步数
                if((i+j)<nums.size())
                    cnt[i+j]=min(cnt[i+j],cnt[i]+1);
            }
        }
        return cnt[nums.size()-1];
    }
};
```

优化之后时间复杂度更低的算法：（理解得比较模糊）

假设f(i)为从起点到i需要跳几步，我们会发现f(i)是单调且连续的（1，1，2，2，2，3，3）这种，不会出现（1，1，2，2，3，3，5）这样

```c++
class Solution {
public:
    int jump(vector<int>& nums) {
        int n=nums.size();
        vector<int> f(n);
        for(int i=1,j=0;i<n;i++){ // i来枚举段
            while(j+nums[j]<i) j++; // 当上一段的距离跳不到i，那就找不到边界，j就一直往后走，找到和i对应的j
            f[i]=f[j]+1; // 因为f是递增的，第一个找到的j就是最小值
        }
        return f[n-1];
    }
};
```









#### 1441.用栈操作构建数组（独立完成）

遍历target数组同时也遍历1-n数字，如果数组中需要的下一个数target[i]不等于t，那就一直t++，执行push pop操作。否则只执行push操作，t也需要+1

```c++
class Solution {
public:
    vector<string> buildArray(vector<int>& target, int n) {
        vector<string> res;
        int t=1;
        for(int i=0;i<target.size();i++){
            while(target[i]!=t){
                res.push_back("Push");
                res.push_back("Pop");
                t++;
            }
            res.push_back("Push");
            t++;
        }
        return res;
    }
};
```



#### 48.旋转图像

把数字矩阵进行顺时针的翻转（直接顺时针反转非常复杂，可以先对角线把左下和右上三角翻转、再左右翻转）

```c++
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n=matrix.size();
        // 先对角线翻转
        for(int i=0;i<n;i++)
            for(int j=0;j<i;j++)
                swap(matrix[i][j],matrix[j][i]);
				
      	// 再左右翻转
        for(int i=0;i<n;i++){
            for(int j=0,k=n-1;j<k;j++,k--)
                swap(matrix[i][j],matrix[i][k]);
        }
    }
};
```





#### 45.跳跃游戏2





#### 55.跳跃游戏

数组中的数字表示从这里出发，最多可以往后跳几步。判断是否可以从起点位置跳到终点位置。

通过推导我们发现能够跳到的位置一定是从起点开始连续的一段（反证法证明）

因此我们找能够跳到的最大位置就是找（i+nums[i]）的最大值

从左往右扫描，记录可以跳到的最大一个位置，然后不断更新这个最大位置，如果在下标i的时候发现前面能够跳到的最大位置j小于i，那么肯定跳不到终点，返回false即可

```c++
class Solution {
public:
    bool canJump(vector<int>& nums) {
        int j=0; // 记录扫描过程中能够跳到的最远距离
        for(int i=0;i<nums.size();i++){
            if(j<i) return false;
            j=max(j,i+nums[i]);
        }
        return true;
    }
};
```



#### 10.正则表达式匹配（非常难！DP问题）

DP可以用循环或者递归来做，都是一样的

· 匹配任意单个字符，*匹配零或者多个前面的元素

边界条件：· * 可以匹配任意字符串

匹配距离：aab可以匹配c\*a\*b（0个c，2个a）

DP的两个要素：状态表示和状态计算

* 状态表示：
  * 集合：s[1-i]和p[1-j]的匹配方案（主要取决于*表示几个字符）
  * 属性：存bool值，表示是否存在一个合法方案
* 状态计算：
  * p[j]不是*时，匹配的情况：（si=pj或者pj=点点）**<u>且</u>**满足s[i-1]和p[j-1]是匹配的**<u>（即f[i-1,j-1]为true)</u>**
  * p[j]是*时，我们需要枚举这个\*表示多少个字符
    * *表示0个字符，那么p中p[j-1]和p[j]的存在都没有意义可以删掉，那么此时匹配的条件就是f(i-1,j-2)为true
    * *表示一个字符，那么匹配的条件是s[i-1]==p[i-1]且f(i-1,j-2)为true
    * *表示2个字符，那么匹配的条件是s[i-1]==s[i-2]==p[i-1]且f(i-2,j-2)为true
    * ...以此类推f(i,j) = f(i,j-2) || f(i-1,j-2) && si与pj匹配 || f(i-2,j-2) && si与pj匹配 && si-1与pj匹配

状态的数量是n平方，转移数量遇到*需要枚举是O(n)，所以整体算法复杂度是n立方，可以优化为整体n平方

优化方式：

f(i-1,j) =  f(i-1,j-2) || f(i-2,j-2) && si-1 || f(i-3,j-2) && si-1 && si-2

列出f(i-1,j)的公式之后，我们会发现f(i-1,j)和f(i,j)的后半段非常相似，所以f(i,j)的计算公式变为：f(i,j) = f(i,j-2) || f(i-1,j) && si与pj-1匹配（又分为si=pj-1或者pj-1是个点）（pj是\*，pj-1才是之前的字符）（这样状态转移的时间复杂度就变成了O(1)，不需要挨个枚举*表示几个字符，总体算法时间复杂度就是n平方

```c++

```





#### 724.寻找数组的中心下标（简单题，独立完成）

```c++
class Solution {
public:
    int pivotIndex(vector<int>& nums) {
        // 计算前缀和，方便进行后面的计算
        for(int i=1;i<nums.size();i++){
            nums[i]+=nums[i-1];
        }

        for(int i=0;i<nums.size();i++){
            if(i==0){
                if(nums[nums.size()-1]-nums[0]==0) return 0;
                continue;
            }
            int l=nums[i-1];
            int r=nums[nums.size()-1]-nums[i];
            if(l==r) return i;
        }
        return -1;
    }
};
```



#### 35.搜索插入位置

假设条件是x>=target,我们搜索的就是满足条件的第一个位置

如果所有数都小于target，那我们找的就是所有数字之后的位置，nums.size( )位置（所以最开始l=0,r=nums.size() )

```c++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
        int l=0,r=nums.size();

        while(l<r){
            int mid=l+r>>1; // 这里不是l+r+1>>1的话就不需要r=nums.size()了
            // 因为>>1是下取整操作，mid肯定小于nums.size()
            if(nums[mid]>=target) r=mid;
            else l=mid+1;
        }
        return l;
    }
};
```



#### 34.在排序数组中查找元素的第一个位置和最后一个位置

重点在于画图判断

```c++
class Solution {
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        if(nums.empty()) return {-1,-1};

        int l=0,r=nums.size()-1;
        // 找到左边边界（开始位置）
        while(l<r){
            int mid=l+r>>1;
            if(nums[mid]>=target) r=mid;
            else l=mid+1;
        }
        // 如果并没有成功找到左边边界
        if(nums[r]!=target){
            return {-1,-1};
        }
				// 找到右边边界（结束位置）
        int L=r;
        l=0,r=nums.size()-1;
        while(l<r){
            int mid=l+r+1>>1;
            if(nums[mid]<=target) l=mid;
            else r=mid-1;
        }
        return {L,r};
    }
};
```



####  36.有效的数独

模拟题，判断当前的状态下是否合法（不用管以后数独是否有解）

分别判断行、列还有小方格（3x3格）

二刷的时候注意memset的位置，每次重新判断一行、列、3x3小方格都需要重新把st置为0

```c++
class Solution {
public:
    bool isValidSudoku(vector<vector<char>>& board) {
        bool st[9];

        // 判断行
        for(int i=0;i<9;i++){
            memset(st,0,sizeof st);
            for(int j=0;j<9;j++){
                if(board[i][j]!='.'){
                    int t=board[i][j]-'1'; // 1-9字符变为0-8
                    if(st[t]) return false;
                    st[t]=true;
                }
            }
        }

        // 判断列
        for(int i=0;i<9;i++){
            memset(st,0,sizeof st);
            for(int j=0;j<9;j++){
                if(board[j][i]!='.'){
                    int t=board[j][i]-'1'; // 1-9字符变为0-8
                    if(st[t]) return false;
                    st[t]=true;
                }
            }
        }

        // 判断小方格
        for(int i=0;i<9;i+=3){
            for(int j=0;j<9;j+=3){
                memset(st,0,sizeof st);
                for(int x=0;x<3;x++){
                    for(int y=0;y<3;y++){
                        if(board[i+x][j+y]!='.'){
                            int t=board[i+x][j+y]-'1';
                            if(st[t]) return false;
                            st[t]=true;
                        }
                        
                    }
                }
            }
        }
        return true;
    }
};
```



 28.找出字符串中第一个匹配项的下标（kmp算法模板）

kmp算法从1开始比较方便，所以最前面补一位

next数组表示所有p[1-i]中相等的前后缀中最长的长度（特殊情况：next[1]=0)

```c++
class Solution {
public:
    int strStr(string s, string p) {
        if(p.empty()) return 0;
        
        int n=s.size(),m=p.size();
        s=' '+s,p=' '+p;

        vector<int> next(m+1); // 因为下标从1开始所以有m=1个数
        // 先求next数组
        for(int i=2,j=0;i<=m;i++){
            while(j && p[i]!=p[j+1]) j=next[j];
            if(p[i]==p[j+1]) j++; // 往后移动一位
            next[i]=j;
        }
        // 匹配的过程
        for(int i=1,j=0;i<=n;i++){
            while(j && s[i]!=p[j+1]) j=next[j]; // 不匹配的时候就回到next[j]（其实next[j]本来表示长度，但是下标从1开始，所以next[j]也恰好是我们要跳到的下标位置）
            if(s[i]==p[j+1]) j++;
            if(j==m) return i-m; // 返回匹配的起始位置
        }
        return -1;
    }
};
```





 940.不同的子序列 II（类似于一个竞赛题，叫做低买）

这是一个DP问题

1.状态表示f(i,j)

f(i,j)表示哪个集合 所有由前i个字母构成，且结尾为j的**不同**方案集合

f(i,j)表示集合的哪个属性（集合的数量）

2.状态计算（注意！这里的ai表示最后一个j，j表示任意一个j都可以）

(1)ai!=j f(i,j)=f(i-1,j)

(2)ai=j f(i,j)=

枚举一下倒数第2个数是什么，全部加在一起就是以ai结尾的方案的和

<img src="/Users/caizy/Desktop/good job/pics/截屏2022-10-14 15.37.46.png" alt="截屏2022-10-14 15.37.46" style="zoom:50%;" />

图中的每个子集就是f(i-1,k)（表示以k结尾的集合数）

通过分析发现ai结尾的所有集合数的和就是j结尾的所有数量的和

```c++
class Solution {
public:
    int distinctSubseqII(string s) {
        // 代码简单但是思路巨难
        const int MOD=1e9+7;
        int f[26]={};

        for(auto c:s){
            int x=c-'a',s=1;
            for(int i=0;i<26;i++){ // 倒数第二个元素的选法
                s=(s+f[i])%MOD;
            }
            f[x]=s;
        }
        int res=0;
        for(int x:f) res=(res+x)%MOD;
        return res;
    }
};
```





 365.水壶问题

做法1:暴搜（枚举每一次的操作，然后看是否可以得到c升水）

做法2:数学方式（推荐）

假设AB两容器的容量分别为a和b

把AB作为一个整体，看AB和外界的水的交换一共有几种情况

操作的过程中，两个杯子的状态不可能同时既不空也不满（至少一个杯子是空或者满），分析如下

1.往一个杯子里倒水（肯定倒满）

2.一个杯子往外倒水（倒空）

3.一个杯子往另一个杯子里倒水（倒到此杯子为空，或者另一个杯子为满）

所以杯子AB作为整体之后，与外界的水的交换只有4种可能（+a,-a,+b,-b）（其他情况分析之后都没有意义，肯定不是最优解）

最后+a和-a抵消，+b和-b抵消之后总共的水肯定是ax+by，看ax+by是否=c

根据裴属定理，=c的充分必要条件是c可以整除ab的最大公约数



能凑出来的话必须c<=a+b,c>=0,且ax+by的两个系数xy必须至少一个大于0，假设x大于0

x>0,y>0 成立;

x>0,y<=0成立;

```c++
class Solution {
public:
    bool canMeasureWater(int a, int b, int c) {
        if(c>a+b) return false;
        return (c==0) | c%gcd(a,b)==0;
    }

    int gcd(int a, int b){
        return b ? gcd(b,a%b):a; 
    }

};
```



 3.无重复字符的最长子串（看了代码才能二刷，错误比较多）

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int> count;
        int res=0;

        for(int i=0,j=0;i<s.size();i++){
            count[s[i]]++;
            while(count[s[i]]>1){
                count[s[j++]]--;
            }
            res=max(res,i-j+1);
        }
        return res;
    }
};
```



 235.二叉搜索树的最近公共祖先

最坏时间复杂度是O(h),每次只递归一个分支

首先，树的问题就看能不能递归来完成

对于二叉树中的两个节点p和q有以下两种情况

1.pq分别在node的左右两侧，那此时node就是最近公共祖先

2.pq在node的左侧，递归去左侧寻找最近公共祖先

3.pq在node的右侧，递归去右侧寻找

![截屏2022-10-04 09.44.12.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-04%2009.44.12.png)

```C++
class Solution {
public:
    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        if(p->val > q->val) swap(p,q); // 为了方便，让p的权是小的
        if(p->val <= root->val && q->val >= root->val) return root; // p和q分别在root的左边和右边，或者p或q就是root（对应值相等的情况），表示root是最近公共祖先
        if(q->val < root->val) return lowestCommonAncestor(root->left,p,q);
      // p和q都在root的左边，递归去root的左子树寻找
        else return lowestCommonAncestor(root->right,p,q);
      // p和q都在root的右边，递归去root的右子树寻找
    }
};
```



 236.二叉树的最近公共祖先

最坏的时间复杂度是O(n)，即把每一个节点都遍历一遍

第二遍看有点看不懂了，为什么要判断root==p这样

```C++
class Solution {
public:
    TreeNode* ans=NULL;

    TreeNode* lowestCommonAncestor(TreeNode* root, TreeNode* p, TreeNode* q) {
        dfs(root,p,q);
        return ans;
    }
    
    int dfs(TreeNode* root, TreeNode* p, TreeNode* q){ // 返回两位的二进式数字来表示root以及root的子树中有没有q和p节点，00表示都没有，01表示有没q有p，以此类推
        if(!root) return 0;
        int state=dfs(root->left,p,q);
        if(root==p) state|=1;
        else if(root==q) state|=2;
        state|=dfs(root->right,p,q);
        if(state==3 && !ans) ans=root; // 表示当前子树中有p和q，且ans没有赋值，是第一个包含p和q的子树
        return state;
    }
};
```



 199.二叉树的右视图

```C++
class Solution {
public:
    vector<int> rightSideView(TreeNode* root) {
        queue<TreeNode*> q;
        vector<int> res;
        if(!root) return res;
        q.push(root);
        while(q.size()){ // 队列不为空的时候
            int len=q.size();
            for(int i=0;i<len;i++){
                auto t=q.front();
                q.pop();
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
                if(i==len-1) res.push_back(t->val);
            }
        }
        return res;
    }
};
```



 101.对称二叉树

先比较两颗子树的根节点是否一样，再比较根节点左右子树是否对称（左边的左子树和右边的右子树是左右对称，左边的右子树和右边的左子树是左右对称），递归一下

```C++
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(!root) return true;
        return dfs(root->left,root->right);
    }

    bool dfs(TreeNode* p, TreeNode* q){
        if(!p && !q) return true;
        if(!p || !q || p->val!=q->val) return false;
        return dfs(p->left,q->right) && dfs(p->right,q->left);
    }
};
```



 102.二叉树的层序遍历

就借助队列就好了

```C++
class Solution {
public:
    vector<vector<int>> levelOrder(TreeNode* root) {
        queue<TreeNode*> q;
        vector<vector<int>> res;

        if(root) q.push(root);
        while(q.size()){
            int len=q.size();
            vector<int> level;
            while(len--){
                auto t=q.front();
                q.pop();
                level.push_back(t->val);
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
            }
            res.push_back(level);
        }
        return res;
    }
};
```



 103.二叉树的锯齿形层次遍历

```C++
 class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        queue<TreeNode*> q;
        if(root) q.push(root);

        int cnt=0;
        while(q.size()){
            vector<int> level;
            int len=q.size();
            while(len--){
                auto t=q.front();
                q.pop();
                if(t->left) q.push(t->left);
                if(t->right) q.push(t->right);
                level.push_back(t->val);
            }
            if(++cnt%2==0) reverse(level.begin(),level.end());
            res.push_back(level);
        }
        return res;
    }
};
```



 114.二叉树展开为链表

（1）存在左儿子，将左子树的右链插入当前点的右边

（2）否则遍历右儿子（root往右边移动）

看似遍历了两遍，其实没有重复遍历，还是O(n)的复杂度

```C++
class Solution {
public:
    void flatten(TreeNode* root) {
        while(root){
            auto p=root->left;
            if(p){ // 找到root左子树的右链
                while(p->right) p=p->right; // 找到root左子树的右链的末尾
                // 把右链拼接到root的左子树上
                p->right=root->right; // root->left是右链的开头
                root->right=root->left; // p是右链的结尾
                root->left=NULL; // 注意拼接之后root就没有左子树了
            }
            root=root->right;
        }
    }
}; 
```



 69.X的平方根

使用二分算法来找出使得y方<x的那个y方

```C++
class Solution {
public:
    int mySqrt(int x) {
        int l=0,r=x;
        while(l<r){
            int mid=l+1ll+r>>1; // 找到mid（之所以先加1ll是怕爆int）
            if(mid<=x/mid) l=mid; // 其实是x<mid*mid 但是怕mid*mid之后爆int
            else r=mid-1;
        }
        return r;
    }
};
```



 141.环形链表

快慢指针做法，O(n)的算法，两个指针所走的步数最多是3x+3(n-x)步

假设总长度为n，直线部分x，不是直线的部分就是n-x

慢指针走到环入口的时候走了x步，那此时快指针走了2x步。他们在环上面相差最大的距离是n-x（就是一整个环的长度）。每走一步快指针就追上慢指针1步，所以快慢指针一共再走3(n-x)步就能相遇。所以一共最多走3x+3(n-x)步，是O(n)的。

![截屏2022-09-29 11.44.29.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-09-29%2011.44.29.png)

（1）有环则会相遇

（2）无环则会走到NULL

```C++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head || !head->next) return false;
        auto s=head,f=head->next; // 这里为什么要错开呢
        while(f){
            s=s->next,f=f->next;
            if(!f) return false;
            f=f->next;
            if(s==f) return true;
        }
    return false;
    }
};
```



 142.环形链表2（找到入环的第一个节点）

首先使用快慢指针做法找到两指针相遇的位置

然后将慢指针指向head，快指针往后移动一位，一直走到相遇就是环的入口位置

sf指针在c点相遇，ab之间为x，bc之间为y

让sf指针在c点同步后退，当慢指针s到了入口点b点时，慢指针后退了y步，快指针则后退2y步来到c‘点。此时，慢指针从a出发走了x步，快指针从a出发走了2x步，因此可以得出，快指针走x步到达b之后又走了x步到达c’点。

从b出发，快指针走x走到c‘，那从c出发的话快指针走x步就能走到b点。

同时，慢指针从a出发，走x步也会走到b点入口点。

那么就首先找到快慢指针的相遇点，然后让s回到起点a，让f在相遇点c的下一位，一起往前走直到相遇即为入口点。

![截屏2022-10-04 10.15.08.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-04%2010.15.08.png)

```C++
class Solution {
public:
    ListNode *detectCycle(ListNode *head) {
        if(!head || !head->next) return NULL;

        auto s=head,f=head->next;
        while(f){
            s=s->next;
            f=f->next;
            if(!f) return NULL;
            f=f->next;
            if(s==f){
                s=head;
                f=f->next;
                while(s!=f) s=s->next,f=f->next;
                return s;
            }
        }
        return NULL;
    }
};
```



 143.重排链表（战略放弃）

首先找到后面一半的链表把后面的指针反转

然后再把前面一半和后面一半两截按照规定的方式拼好

这个题比较麻烦

```C++
class Solution {
public:
    void reorderList(ListNode* head) {
        if(!head) return;
        int n=0;
        for(auto p=head;p;p=p->next) n++; // 先找到尾节点
        auto mid=head;

        // 找到一半的节点
        for(int i=0;i<(n+1)/2-1;i++) mid=mid->next;
        auto a=mid,b=a->next;

        // 反转后面一半
        for(int i=0;i<n/2;i++){
            auto c=b->next;
            b->next=a,a=b,b=c;
        }

        auto p=head,q=a;

        // 开始把前后两半链表交错起来
        for(int i=0;i<n/2;i++){
            auto o=q->next;
            q->next=p->next;
            p->next=q;
            if(n%2==0 && i==n/2-1) q->next=NULL;
            p=q->next,q=o;
        }
        if(n%2) p->next=NULL;
    }
};
```



 82.删除排序链表中的重复元素2

有重复的话所有这个数字都会被删掉

```C++
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        auto dummy=new ListNode(-1);
        dummy->next=head;

        auto p=dummy;
        while(p->next){ // 其实p->next才是当前节点，p是上一个节点
            auto q=p->next->next;
            while(q && q->val==p->next->val) q=q->next;
            if(p->next->next==q) p=p->next; // 证明这个数只有一个，继续
            else p->next=q; // 这个数有重复，删掉所有这个数
        }
        return dummy->next;
    }
};
```

  

 1171.从列表中删除总和值为零的连续节点

```C++

```



 2.两数相加

两个链表，每个节点存放了一位数字，将两个链表所表示的数字相加然后返回一个表示数字之和的链表

```C++
class Solution {
public:
    ListNode* addTwoNumbers(ListNode* l1, ListNode* l2) {
        auto dummy=new ListNode(-1),tail=dummy;
        int t=0; // 进位
        while(l1 || l2 || t) // l1,l2没有循环完，或者是进位不为0的时候
        {
            if(l1) t+=l1->val,l1=l1->next;
            if(l2) t+=l2->val,l2=l2->next;
            // 创造新的节点，连接到链表的尾部
            tail=tail->next=new ListNode(t%10);
            t/=10;
        }
        return dummy->next;
    }
};
```



 138.复制带随机指针的链表

```C++

```



 18.四数之和

比较复杂的n重循环



 287.寻找重复数

类似于lc 14题

**可惜听不懂这在干啥**

```C++
// 快慢指针做法 听不懂视频
class Solution {
public:
    int findDuplicate(vector<int>& nums) {
        int a=0,b=0;
        while(true){
            a=nums[a];
            b=nums[nums[b]];
            if(a==b){
                a=0;
                while(a!=b){
                    a=nums[a];
                    b=nums[b];
                }
                return a;
            }
        }
        return -1;
    }
};
```



 53.最大子序和（自己独立完成的！）

```C++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int r=nums[0],c=1;
        // r表示存的数字，c表示这个数字的count（负责计数的）
        for(int i=1;i<nums.size();i++)
        {
            if(nums[i]==r) c++;
            else{
                if(c==0){
                    r=nums[i];
                    c=1;
                }
                else c--;
            }
        }
        return r;
    }
};
```



 169.多数元素

```C++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int r=nums[0],c=1;
        // r表示存的数字，c表示这个数字的count（负责计数的）
        for(int i=1;i<nums.size();i++)
        {
            if(nums[i]==r) c++;
            else{
                if(c==0){
                    r=nums[i];
                    c=1;
                }
                else c--;
            }
        }
        return r;
    }
};
```



 88.合并两个有序数组（自己独立完成的！）

```C++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int a=m-1;
        int b=n-1;
        for(int i=m+n-1;i>=0;i--)
        {
            if(a>=0 && b>=0)
            {
                if(nums1[a]>=nums2[b]) nums1[i]=nums1[a--];
                else nums1[i]=nums2[b--];
            }
            else if(a>=0) break;
            else nums1[i]=nums2[b--];
        }
}
};
```



 121.买卖股票的最佳时机

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n[100000];
        int maxp=-1;
        for(int i=prices.size()-1;i>=0;i--)
        {
            if(i==prices.size()-1) n[i]=prices[i];
            n[i]=max(maxp,prices[i]);
            maxp=n[i];
        }
       int res=0;
       for(int i=0;i<prices.size()-1;i++){
           if(prices[i]<n[i+1])
           {
               res=max(res,n[i+1]-prices[i]);
           }
       }
       return res;
    }
};
```



 14.最长公共前缀

```C++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        string res;
        if(strs.empty()) return res;

        for(int i=0;;i++){
            if(i>=strs[0].size()) return res;
            char c=strs[0][i];

            for(auto& str:strs){
                if(str.size()<=i || str[i]!=c) return res;
            }
            res+=c;
        }
        return res;
    }
};
```



10月1日

 122.买卖股票的最佳时机2

一次跨度大于1天的交易可以拆成多个1天的交易，所以这道题只需要首先计算出所有跨度1天的差额，然后在所有1天的差额中选择正数相加就可以了

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int cnt[100001];

        for(int i=0;i<prices.size()-1;i++)
        {
            cnt[i]=prices[i+1]-prices[i];
            cout<<cnt[i]<<" ";
        }
        int res=0;

        for(int i=0;i<prices.size()-1;i++)
        {
            if(cnt[i]>0) res+=cnt[i];
        }
        return res;
    }
};
```



剑指Offer 53.在排序数组中查找元素1

```C++

```



 31.下一个排序

实现next_permutation函数

1.找到数组末尾降序序列a，取降序序列第一个数x（或者说是升序和降序序列的分界线）（如果没有降序序列，就把最后一个数看作是长度为1的降序序列）

2.找到分界点x前面的第一个数y（这个数肯定小于x），把y和降序序列中大于y的最小的数z交换，然后重新排列末尾降序序列a为升序（在代码中就是reverse反转一下，因为本来是降序，反转之后就是升序了）

3.特殊情况！！如果整个数组都是降序的，那么就重新排列一下为全升序（reverse）

```C++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        int k=nums.size()-1;
        while(k>0 && nums[k-1]>=nums[k]) k--; // 如果一直是升序、相等情况就继续往后
        // 找到开始降序的地方

        cout<<"开始降序的位置是"<<nums[k]<<endl;
        if(k<=0) // 整个序列都是降序
        {
            reverse(nums.begin(),nums.end());
        }
        else // 要找到第一个小于当前数的数
        {
            int t=k;
            while(t<nums.size() && nums[t]>nums[k-1]) t++;
            // 找到降序序列中大于开始降序分界点前一个数字的最小数字
            cout<<"swap的两个数字分别是"<<nums[t-1]<<" "<<nums[k-1];
            swap(nums[t-1],nums[k-1]);
            reverse(nums.begin()+k,nums.end());
        }
    }
};
```



 33.搜索旋转排序数组**（二分算法背模板）**

本来是全部升序的序列，现在砍成两段把前一段拼接到后面去

所以区分前后序列的性质就是大于还是小于nums[0]数组，因此可以使用二分算法

```C++
class Solution {
public:
    int search(vector<int>& nums, int target) {
        if(nums.empty()) return -1;

        int l=0,r=nums.size()-1;

        // 首先用二分算法找到从哪里开始翻转
        while(l<r){
            int mid=l+r+1>>1;
            if(nums[mid]>=nums[0]) l=mid;
            else r=mid-1;
        }
        
        // 判断要搜索的数
        if(target>=nums[0]) l=0; // 说明target在第一个区间里
        else l=r+1,r=nums.size()-1; // 说明target在第二个区间里

        // 然后用二分算法寻找这个数字
        while(l<r){
            int mid=l+r>>1;
            if(nums[mid]>=target) r=mid;
            else l=mid+1;
        }
        if(nums[r]==target) return r;
        return -1;
    }
};
```



 104.二叉树的最大深度

```C++
class Solution {
public:
    int maxDepth(TreeNode* root) {
        if(!root) return 0;
        return max(maxDepth(root->left),maxDepth(root->right))+1;
    }
};
```



 105.从前序与中序遍历序列

```C++
class Solution {
public:
    unordered_map<int,int> pos;
    TreeNode* buildTree(vector<int>& preorder, vector<int>& inorder) {
        for(int i=0;i<preorder.size();i++) pos[inorder[i]]=i;
        return build(preorder,inorder,0,preorder.size()-1,0,preorder.size()-1);
    }

    TreeNode* build(vector<int>& preorder, vector<int>& inorder, int pl, int pr, int il, int ir)
    {
        if(pl>pr) return NULL;

        auto root= new TreeNode(preorder[pl]);
        int k=pos[root->val];

        root->left=build(preorder,inorder,pl+1,pl+1+k-1-il,il,k-1);
        root->right=build(preorder,inorder,pl+1+k-1-il+1,pr,k+1,ir);
        return root;
    }
};
```



 209.长度最小的子数组

使用双指针算法，因为是单向的问题

```C++
class Solution {
public:
    int minSubArrayLen(int target, vector<int>& nums) {
        int res=INT_MAX;
      	// i是前面的指针，j是后面的指针
        for(int i=0,j=0,sum=0;i<nums.size();i++)
        {
            sum+=nums[i];
            while(sum-nums[j]>=target) sum-=nums[j++];

            if(sum>=target) res=min(res,i-j+1);
        }
        if(res==INT_MAX) res=0;
        return res;
    }
};
```



 98.验证二叉搜索树（看了解答后二刷成功）

这里的t分别记录：t[0]表示以这个节点为根的树是否是二叉搜索树，t[1]表示当前已经遍历的节点中的最小值，t[2]表示当前的最大值

```C++
class Solution {
public:
    bool isValidBST(TreeNode* root) {
        if(!root) return true;
        return dfs(root)[0];
    }
    
    vector<int> dfs(TreeNode* root)
    {
        vector<int> res({1,root->val,root->val});

        if(root->left)
        {
            auto t=dfs(root->left); // 先去递归遍历左子树
            if(!t[0] || t[2]>=root->val) res[0]=0; // 如果左子树的最大值大于root值表示不是排序树
            // 更新最大最小值
            res[1]=min(res[1],t[1]);
            res[2]=max(res[2],t[2]);
        }

        if(root->right)
        {
            auto t=dfs(root->right); // 递归遍历右子树
            if(!t[0] || t[1]<=root->val) res[0]=0; // 如果右子树的最小值小于root值表示不是排序树
            res[1]=min(res[1],t[1]);
            res[2]=max(res[2],t[2]);
        }
        return res;
    }
};
```



 234.回文链表（自己独立完成！）

找到链表的1/2分界点，然后反转后面一半的链表。

反转之后，使一个指针在头，一个指针在1/2处同时往后走，看后一个指针是否能走到末尾，如果可以就是回文

```C++
// 我自己的做法
class Solution {
public:
    bool isPalindrome(ListNode* head) {
        if(!head || !head->next) return true;
        auto node=head;
        int cnt=0;
        while(node)
        {
            node=node->next;
            cnt++;
        }
        node=head;
        auto pre=new ListNode(-1);
        pre->next=head;
        for(int i=0;i<cnt/2;i++ )
        {
            node=node->next;
            pre=pre->next;
        }
        if(cnt%2==0)
        {
            pre->next=NULL;
        }
        else
        {
            pre->next=NULL;
            node=node->next;
        }
        pre=node;
        auto t=node->next;
        node->next=NULL;
        node=t;

        while(node)
        {
            if(!node->next)
            {
                node->next=pre;
                pre=node;
                break;
            }
            else
            {
                auto t=node->next;
                node->next=pre;
                pre=node;
                node=t;
            }
        }
        node=head;

        while(node && pre && node->val==pre->val){
            cout<<node->val<<" "<<pre->val<<" ";
            node=node->next,pre=pre->next;
        }
       
        if(node==NULL && pre==NULL) return 1;

        return 0;
    }
};
```



```C++
// y总的做法

class Solution {
public:
    bool isPalindrome(ListNode* head) {
        int n=0;
        for(auto p=head;p;p=p->next) n++;

        if(n<=1) return true;

        int half=n/2;
        auto a=head;

        for(int i=0;i<n-half;i++) a=a->next;
        
        auto b=a->next;

        for(int i=0;i<half-1;i++){
            auto c=b->next;
            b->next=a;
            a=b,b=c;
        }

        auto p=head,q=a;
        bool success=true;

        for(int i=0;i<half;i++){
            if(p->val != q->val){
                success = false;
                break;
            }
            p=p->next;
            q=q->next; 
        }

        auto tail=a;
        b=a->next;

        // 链表恢复原状
        for(int i=0;i<half-1;i++){
            auto c=b->next;
            b->next=a;
            a-b,b=c;
        }
        tail->next=NULL;
        
        return success;

    }
};

```



 958.二叉树的完全性检验（复习后二刷一遍过）

```C++
class Solution {
public:
    int n=0; // n表示点的个数
    int p=0; // 节点编号的最大值

    bool dfs(TreeNode* root, int k) // k表示当前节点的编号
    {
        if(!root) return true;
        if(k>100) return false;
        n++;
        p=max(p,k);
        return dfs(root->left,k*2) && dfs(root->right,k*2+1);
    }

    bool isCompleteTree(TreeNode* root) {
        if(!dfs(root,1)) return false;
        return n==p;  // 节点的个数=编号最大值，即每个编号都有一个正确位置的点对应
    }
};
```



 128.最长连续序列（用哈希表来枚举）

每次找到枚举的开头（它在hash中存在，且比它小1的数字不存在）

然后往后找连续的数字，如果更大就更新一下res

```C++
class Solution {
public:
    int longestConsecutive(vector<int>& nums) {
        unordered_set<int> S;
        for(auto x:nums) S.insert(x);

        int res=0;
        for(auto x:nums){
            if(S.count(x) && !S.count(x-1)){ // 表示可能是一段连续序列的开头
                int y=x; // 从x开始枚举
                S.erase(x); // 删除已经枚举过的，这样时间复杂度才能保证是O(n)的
                while(S.count(y+1)){
                    y++;
                    S.erase(y); // 同上，删除已经枚举的
                }
                res=max(res,y-x+1);
            }
        }
        return res;
    }
};
```



 974.和可被K整除的子数组

求以i为最后一个数的子数组中，最多能有几个子数组被K整除

这个时候我们就可以枚举子数组的开头，假设为j。求一段连续子数组的和可以用前缀和的思想。

求以j结尾的子数组中有多少个可以被K整除，假设子数组开头为j

那么从j加到i的合就是s[i]-s[j-1]（s表示前缀和），s[i]-s[j-1]能被K整除意味着s[i]和s[j-1]是同余的

那么问题就转化为，每次有一个末尾节点i的时候，判断前面有多少个前缀和与当前s[i]同余，因此开一个cnt数组来记录同余

```C++
class Solution {
public:
    int subarraysDivByK(vector<int>& nums, int k) {
        int n=nums.size();
        vector<int> s(n+1);

        for(int i=1;i<=n;i++) s[i]=s[i-1]+nums[i-1]; // 之所以是nums[i-1]是因为s的下标从1开始，nums从0开始
        unordered_map<int,int> cnt;

        cnt[0]++; // 初始的时候j可以取1，那么j-1就可以取0，s[0]是等于0的，所以cnt[0]++
        int res=0;

        for(int i=1;i<=n;i++){
            int r=(s[i]%k+k)%k; // s[i]有可能是负数，为了统一正负数余数所以多处理了一下
            res+=cnt[r]; // 看一下在nums[i]之前，与nums[i]同余数的有几个（不包括nums[i]本人）
            cnt[r]++; // 让cnt[r]+1
        }
        return res;
    }
};
```



 79.单词搜索（暴搜）

就用dfs搜索就好

```C++
class Solution {
public:
    bool exist(vector<vector<char>>& board, string word) {
        for(int i=0;i<board.size();i++){
            for(int j=0;j<board[i].size();j++){
                if(dfs(board,word,0,i,j)) return true; // 从每一个字母开始往后寻找
            }
        }
        return false;
    }

    int dx[4]={-1,0,1,0},dy[4]={0,1,0,-1};

    bool dfs(vector<vector<char>>& board, string& word, int u, int x, int y)
    {
        if(board[x][y]!=word[u]) return false; // 对应位置不是我们要找的下一个字母，返回
        if(u==word.size()-1) return true; // 已经找到了最后一个数字

        // 记录这个字母，方便遍历之后恢复原样
        char t=board[x][y];
        // 字母改为标记字符，这样不会重复走过
        board[x][y]='.';

        for(int i=0;i<4;i++){
            int a=x+dx[i],b=y+dy[i];
            if(a<0 || a>=board.size() || b<0 || b>=board[0].size() || board[a][b]=='.') continue;
            if(dfs(board,word,u+1,a,b)) return true;
        }
        board[x][y]=t; // 恢复原来的字母
        return false;
    }
};
```



 162.寻找峰值

```C++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        // 因为每次可以排除掉一半，所以通过二分来做
        int l=0,r=nums.size()-1;
        while(l<r){
            int mid=l+r>>1;
            if(nums[mid]>nums[mid+1]) // 说明从l-mid都有可能是答案
                r=mid;
            else l=mid+1;
        }
        return r; // 或者l也可以
    }
};
```



10月4日

---

 146.LRU缓存机制（感觉很难只靠自己就写出）

在O(1)的时间实现插入和删除方法

每次需要

1.找到最近最少使用的节点（要根据时间排序）并且以O(1)的时间删除这个节点

2.可以O(1)的时间插入一个节点（每次插入之后排到队列/双链表的最右侧），并判断一下容量是否超过

3.可以O(1)的时间访问一个节点，访问之后还需要把这个节点放到队列的最左侧（左右都可以，看具体怎么规定）

使用**双向链表和哈希表**

```C++
class LRUCache {
public:
    struct Node{
        int key, val;
        Node* left,*right;
        Node(int _key, int _val):key(_key),val(_val),left(NULL),right(NULL){}
    }*L,*R;

    unordered_map<int,Node*> hash; 
    int n;

    LRUCache(int capacity) {
        n=capacity;
        // L和R是左右侧的头节点 类似dummy的作用 不需要赋值
        L=new Node(-1,-1),R=new Node(-1,-1);
        L->right=R,R->left=L;
    }
    
    void remove(Node* p){
        p->right->left=p->left;
        p->left->right=p->right;
    }

    void insert(Node* p){
        p->right=L->right;
        p->left=L;
        L->right->left=p;
        L->right=p;
    }

    int get(int key) {
        if(hash.count(key)==0) // 哈希表中不存在这个值
            return -1;
        // 一旦这个数被访问过，就需要将这个节点抽出来放到最左边（删除然后重新添加）
        auto p=hash[key];
        remove(p);
        insert(p);
        return p->val;
    }
    
    void put(int key, int value) {
        if(hash.count(key)){ // 有对应的key值在表中
            auto p=hash[key];
            p->val=value; // 修改值为value
            remove(p);
            insert(p); // 重新加入链表
        } else{ 
            if(hash.size()==n){ // 没有对应的key值，需要插入，但是满了
                // 删除最右边的点（R左侧的点）
                auto p=R->left;
                // 从双链表和哈希表中删除
                remove(p);
                hash.erase(p->key); // 把key值从哈希表里移除
                delete p;
            }
            // 添加新的key:value对
            auto p=new Node(key,value);
            hash[key]=p;
            insert(p);
        }
    }
};
```



 147.对链表进行插入排序（二刷，还算顺利）

每次拿到一个节点，找到已排序链表中第一个比它大的节点，然后插入链表

```C++
class Solution {
public:
    ListNode* insertionSortList(ListNode* head) {
        auto dummy=new ListNode(-1); // 新开一条链表，添加虚拟头节点方便操作
        for(auto p=head;p;){
            auto cur=dummy,next=p->next; // 把当前节点的下一个节点记录一下
            while(cur->next && cur->next->val<=p->val) cur=cur->next;
            // cur要么是尾节点，要么是小于p->val的最大节点（反正p就要接在cur节点后）
            p->next=cur->next;
            cur->next=p;
            p=next; // p节点往后移动一位
        }
        return dummy->next;
    }
};
```



 24.两两交换链表中的节点（二刷，指针有点混乱）

头节点很有可能会变，所以我们可以设置一个虚拟头节点

画个图

```C++
class Solution {
public:
    ListNode* swapPairs(ListNode* head) {
        auto dummy=new ListNode(-1);
        dummy->next=head;
        for(auto p=dummy;p->next && p->next->next;)
        {
          	// p表示要交换的两个节点前面的节点，a是第一个点，b是第二个点
            auto a=p->next,b=a->next;
            auto next=b->next;
            p->next=b;
            a->next=b->next;
            b->next=a;
            p=a; // 交换后的a点是两个节点中的后一个节点
        }
        return dummy->next;
    }
};
```



 23.合并k个排序链表（比较难的链表题）（二刷）

与合并两个排序链表的操作类似，合并两个排序链表时就用两个指针分别指向链表的开头（最小值），然后把这两个值中小一点的选出来

在本题中我们用heap最小堆来维护这K个指针（C++当中是优先队列）

```C++
class Solution {
public:
    struct CmP{ // 比较少见，背过就好
        bool operator() (ListNode* a, ListNode* b){
            return a->val>b->val;
        }
    };

    ListNode* mergeKLists(vector<ListNode*>& lists) {
        priority_queue<ListNode*,vector<ListNode*>,CmP> heap;
        // 排序之后的链表的头和尾节点
        auto dummy=new ListNode(-1),tail=dummy;
        // 把所有的k个链表的头节点都放入heap中
        for(auto l:lists) if(l) heap.push(l);
        while(heap.size()) // 堆当中有元素的时候
        {
            // 弹出堆顶元素（即为k个链表的k个头指针指向的元素中最小的一个）
            auto t=heap.top();
            heap.pop();
            // 把当前节点插入到尾节点的后面
            tail->next=t;
            // 插入之后t就是新的尾节点，所以更新一下t
            tail=t;
            // 上面这两句也可以简化为tail=tail->next=t;

            if(t->next) heap.push(t->next);
        }   
        return dummy->next;
    }
};
```



 42.接雨水（困难）(勉强2刷）

注意划分接雨水区域如何计算，变为不同块之间的高度、宽度关系，last用来记录上一个块的高度，栈当中是递减的序列

使用单调栈，维护一个元素高度始终递减的栈，每次先判断是否需要弹出，

```C++
class Solution {
public:
    int trap(vector<int>& height) {
        // 使用单调栈来完成这道题
        stack<int> stk;
        int res=0;
        for(int i=0;i<height.size();i++){
            int last=0; // 注意last每一遍都要重新从0开始
            while(stk.size() && height[stk.top()]<=height[i])
            { // 栈非空且栈顶元素小于等于当前元素
            res+=(height[stk.top()]-last)*(i-stk.top()-1);
              // 高度
            last=height[stk.top()];
            stk.pop();
            }
            if(stk.size()) res+=(height[i]-last)*((i-stk.top()-1));
            stk.push(i);
        }
        return res;
    }
};
```



补充题 排序奇升偶降链表

```C++

```



 32.最长有效括号（思路非常奇特且不具有普遍性，有一部分代码无法理解）

合法括号的两个要求

1.左括号=右括号数量

2.在任何前缀中左括号数量大于等于右括号数量

首先把整个长长的括号序列切开分为很多段（切开的分界是一段序列的第一个不满足左括号数量大于等于右括号数量的右括号），我们寻找的答案一定在切开之后的各个段里（y总反证法证明）

在每一段中枚举右括号，来找距离该右括号最远的满足合法序列的左括号，分情况看弹出之后栈是否为空

start+1表示新的一段的开始下标,start表示上一段的末尾

```C++
class Solution {
public:
    int longestValidParentheses(string s) {
        stack<int> stk;
        int res=0;

        for(int i=0,start=-1;i<s.size();i++){
            if(s[i]=='(') stk.push(i);
            else{
                if(stk.size()){
                    stk.pop(); // 弹出左括号与这个括号匹配
                    if(stk.size()){
                        // 栈不为空，那么最左的一个括号是目前栈顶元素的下一个元素（无法理解）
                        res=max(res,i-stk.top());
                    } else{ 
                        // 如果匹配完这个右括号之后就空了，那么意味着这一整段都是合法的
                        // 这一段的开始是start+1(start为上一段的末尾)
                        res=max(res,i-(start+1)+1);
                    }
                } else{
                    start=i;
                }
            }
        }
        return res;
    }
};
```



 25.K个一组翻转链表（勉勉强强二刷）

注意翻转的时候不仅仅K个节点一组需要翻转，还要在外部拼上这个翻转后的链表

```C++
class Solution {
public:
    ListNode* reverseKGroup(ListNode* head, int k) {
        auto dummy=new ListNode(-1);
        dummy->next=head;

        for(auto p=dummy;;){
            auto q=p;
            for(int i=0;i<k && q;i++) q=q->next;
            if(!q) break; // 如果q变为空，表示后面不足k个元素
           
            auto a=p->next,b=a->next; // a,b用来翻转链表
            for(int i=0;i<k-1;i++)
            {
                // 在k个节点内部翻转
                auto c=b->next;
                // 这里要特别注意，只需要翻转a->b的一个指针，然后ab共同往后移一位
                b->next=a;
                a=b,b=c;
            }
            // 在k个节点的外部把翻转后的链表接到长链表中
            auto c=p->next;
            p->next=a,c->next=b;
            p=c;
        }
        return dummy->next;
    }
};
```



 5.最长回文子串（二刷）（三刷忘记了substr的边界，不够熟练）

```C++
class Solution {
public:
    string longestPalindrome(string s) {
        string res;
        for(int i=0;i<s.size();i++){
            // 回文串长度为奇数情况
            int l=i-1,r=i+1;
            while(l>=0 && r<s.size() && s[l]==s[r]) l--,r++;
            if(r-l-1>res.size()) res=s.substr(l+1,r-l-1);
			// 回文串长度为偶数情况
            l=i,r=i+1;
            while(l>=0 && r<s.size() && s[l]==s[r]) l--,r++;
            if(r-l-1>res.size()) res=s.substr(l+1,r-l-1);
        }
        return res;
    }
};
```





### 10月5日

 41.缺失的第一个正数（二刷一遍过）

做法1：可以先将数组排序，然后从小到大遍历找到缺失的第一个正数，但是排序的时间复杂度是O(logn)稍微有点大

做法2（我们采用的做法）：将数组用一个哈希表存放，然后从1开始枚举，找到缺失的第一个正整数

```C++
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        unordered_set<int> hash;
        for(auto num:nums) hash.insert(num);

        int res=1;
        while(hash.count(res)) res++;
        return res;
    }
};
```



 402.移掉K位数字（经典的贪心题目）

从前往后枚举每一位，分情况判断（这一位要删和这一位不删两种情况）

假设已经枚举了i-1位，现在枚举到第i位，根据si的情况来判断si-1位是否要删除

1.s[i-1]>s[i] 要删掉s[i-1]

2.s[i-1]<s[i] 不能删s[i-1]（不然就留下更大的s[i]）

3.s[i-1]=s[i] 先不删掉s[i-1]，反正s[i]和s[i-1]大小相等 在后面再去判断删不删s[i]是一样的效果

总结就是s[i-1]大于s[i]的时候才要删（在代码中就是res.back( )<c的时候，将res中大于当前c的数字全部删除）

如果到最后还没有删到k位（多了一些数字），直接从后往前删就好，因为最后剩下的序列是递增的序列，末尾的数字都是最大的，直接删掉最后就得到最优解

不需要暴力搜索，因为每次都可以判断出哪一种选择更好，所以每次选择更好的选择即可

![截屏2022-10-05 10.57.58.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-05%2010.57.58.png)



```C++
class Solution {
public:
    string removeKdigits(string num, int k) {
        k=min(k,(int)num.size()); // 缩小一下k
        string res; // 表示最后的结果（剩下的数字）
        for(auto c:num){
            while(k && res.size() && res.back()>c) 
            // 当k>=0表示还有字符需要删 res.size()表示res中有数字（即s[i-1]存在 res.back()>c表示答案的最后一个字符大于c（即s[i-1]>s[i])
           {
               // 删除s[i-1]
                k--;
                res.pop_back();
           }
           res+=c;
        }
        while(k--) res.pop_back(); // 没有删完的话把后缀全部删掉

        // 最后这四行的作用是删除前导0，如果发现全都是0，就将res置为0
        k=0;
        while(k<res.size() && res[k]=='0') k++;
        if(k==res.size()) res+='0'; // 和后面的substr配合使用，将返回值置为0
        return res.substr(k);
    }
};
```



 739.每日温度（经典单调栈的模板题）（磕磕绊绊二刷）

可以去看对应的单调栈模板题（acwing 830.单调栈 找到左边第一个比它小的数字）

本题是找到右边第一个比它大的数字，所以从右往左遍历，如果当前元素大于等于栈顶元素的值就pop( )

```C++
class Solution {
public:
    vector<int> dailyTemperatures(vector<int>& T) {
        vector<int> res(T.size());
        stack<int> stk;
        for(int i=T.size()-1;i>=0;i--){
            // 栈顶元素（比当前数字下标更大，日期更往后的数字）小于等于当前数字,则pop
            while(stk.size() && T[i]>=T[stk.top()]) stk.pop();
            // 栈顶元素是日期更靠后的，第一个大于当前T[i]的数字
            if(stk.size()) res[i]=stk.top()-i;
            stk.push(i);
        }
        return res;
    }
};
```



acwing 830.单调栈

```C++
#include<iostream>
#include<stack>
#include<vector>

using namespace std;

int main()
{
    int n;
    int T[100010];
    int res[100010];
    scanf("%d",&n);
    
    for(int i=0;i<n;i++) cin>>T[i];

    stack<int> stk;
    
    for(int i=0;i<n;i++)
    {
        while(stk.size() && T[stk.top()]>=T[i]) stk.pop();
        if(stk.size()) res[i]=T[stk.top()];
        else res[i]=-1;
        stk.push(i);
    }
    
    for(int i=0;i<n;i++) cout<<res[i]<<" ";
    
    return 0;
}
```



 85.最大矩形（使用单调栈）



 221.最大正方形（使用dp（勉强二刷成功，忘记min之后+1了）

f(i,j)表示以(i,j)为右下角点的最大正方形的边长

假设以点(i,j)为右下角的正方形如图，我们可以得到f(i-1,j), f(i,j-1), f(i-1,j-1) 还有f(i,j)之间的关系如下

1.f(i-1,j)≥f(i,j)-1

2.f(i,j-1)≥f(i,j)-1

3.f(i-1,j-1)≥f(i,j)-1

由此推出f(i,j) ≤ min( f(i-1,j)+1 , f(i,j-1)+1 , f(i,j)+1 ) 

![截屏2022-10-05 14.13.56.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-05%2014.13.56.png)

我们想要证明的是f(i,j)=min( f(i-1,j)+1 , f(i,j-1)+1 , f(i-1,j-1)+1 ) ,所以我们还需要证明f(i,j)≥ min( f(i-1,j)+1 , f(i,j-1)+1 , f(i-1,j-1)+1 ) 

证明方法（反证法）

假设f(i,j)<min( f(i-1,j)+1 , f(i,j-1)+1 , f(i-1,j-1)+1 ) ,那么就会有

f(i-1,j)+1>f(i,j), f(i,j-1)+1>f(i,j) 以及 f(i-1,j-1)+1>f(i,j) 

所以可以推导出

f(i-1,j)≥f(i,j), f(i,j-1)≥f(i,j) 以及 f(i-1,j-1)≥f(i,j)

在图上画出来是下面的样子：

![截屏2022-10-05 14.27.48.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-05%2014.27.48.png)

所以以(i,j)为右下点的最大正方形应该还要更大1位，与原来的假设矛盾，因此假设不成立。反证法证明成功。f(i,j)=min( f(i-1,j)+1 , f(i,j-1)+1 , f(i-1,j-1)+1 )。

```C++
class Solution {
public:
    int maximalSquare(vector<vector<char>>& matrix) {
        if(matrix.empty() || matrix[0].empty()) return 0;
        int n=matrix.size(), m=matrix[0].size();
        vector<vector<int>> f(n+1,vector<int>(m+1));
        
        int res=0;

        for(int i=1;i<=n;i++)
            for(int j=1;j<=m;j++)
                // 这里之所以写matrix[i-1][j-1]是因为坐标的起点不同
                if(matrix[i-1][j-1]=='1'){ // 只有等于1的点要开始
                    f[i][j]=min(f[i-1][j],min(f[i][j-1],f[i-1][j-1]))+1;
                    res=max(res,f[i][j]);
                }
        return res*res;
    }
};
```



 300.最长递增子序列（独立完成，没看教程）（完美二刷）

```C++
class Solution {
public:
    int lengthOfLIS(vector<int>& nums) {
        vector<int> count(nums.size());
        int res=1;

        for(int i=0;i<nums.size();i++){
            count[i]=1;
            for(int j=0;j<i;j++)
                if(nums[j]<nums[i]) count[i]=max(count[j]+1,count[i]);
            res=max(res,count[i]);
        }
        
        return res;
    }
};
```



 46.全排列（独立完成）

```C++
class Solution {
public:
    vector<vector<int>> res;
    bool used[10];
    vector<vector<int>> permute(vector<int>& nums) {
        vector<int> p;
        dfs(nums,p,0);
        return res;
    }

    void dfs(vector<int>& nums, vector<int>& perm, int u){
        if(u==nums.size()) res.push_back(perm);
        for(int i=0;i<nums.size();i++){
            if(used[i]==0){
                perm.push_back(nums[i]);
                used[i]=1;
                dfs(nums,perm,u+1);
                // 记得恢复现场！
                used[i]=0;
                perm.pop_back();
            }
        }
    }
};
```



 4.寻找两个正序数组的中位数

给定两个大小分别为 m 和 n 的正序（从小到大）数组 nums1 和 nums2。请你找出并返回这两个正序数组的 中位数 。

算法的时间复杂度应该为 O(log (m+n)) 

方法1 数组合并一起然后sort

方法2 log(n+m) 递归来做 推荐！

方法3 log(min ( n,m )) 二分算法，细节非常繁琐，不推荐



思路：

假设要找到A和B中从大到小排序的第k个数，先不需要把AB放在一起排序，取A，B数组的k/2位置的数根据大小进行分类讨论

1.A[k/2]<B[k/2]

![截屏2022-10-05 15.39.42.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-05%2015.39.42.png)

由于A[k/2]小于B[k/2]，所以加入把A[k/2]放在全部数字中的排行肯定小于k/2+k/2=k,所以A数组中在A[k/2]左侧的数字的排行更加小于k，所以A左侧舍弃

2.A[k/2]>B[k/2] 同理，B的左侧舍弃

3.A[k/2]=B[k/2] 此时A[k/2]或者B[k/2]就是我们要找的第k个数

```C++
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int tot=nums1.size()+nums2.size();
        if(tot%2==0){
            int left=find(nums1,0,nums2,0,tot/2);
            int right=find(nums1,0,nums2,0,tot/2+1);
            return (left+right)/2.0;
        } else {
            return find(nums1,0,nums2,0,tot/2+1);
        }
    }

    int find(vector<int>& nums1, int i, vector<int>& nums2, int j, int k){
        if(nums1.size()-i>nums2.size()-j) return find(nums2,j,nums1,i,k);

        if(k==1){ // 找第一个数的情况
            // 如果这个较短的nums1为空，只能在nums2中找了
            if(nums1.size()==i) return nums2[j];
            else return min(nums1[i],nums2[j]);
        }
        if(nums1.size()==i) // 如果第一个数组为空
           return nums2[j+k-1]; 
        
        int si=min((int)nums1.size(),i+k/2),sj=j+k-k/2; // si和sj其实是第k/2个元素的下一个元素 所以在使用的时候需要si-1和sj-1
        if(nums1[si-1]>nums2[sj-1])
            return find(nums1,i,nums2,sj,k-(sj-j)); // sj-j表示在nums2中删除的个数
        else return find(nums1,si,nums2,j,k-(si-i));

    }
};
```



#### 10月6日

 148.排序链表（利用归并排序，每次合并两段链表）

```C++

```



 123.买卖股票的最佳时机3（前后缀分解法）

最多只能进行两笔交易（也可以进行一笔交易）

可以用DP来做（不仅是两笔交易，K笔交易也可以），但是y总在这里没用DP问题

枚举两次交易的分界点，枚举的是第二次交易的买入时间i

第一次交易必然在1-i-1中，第二次交易在i以及i后，两段的时间完全独立。如果想要两次交易的总和最大，只需要让前面和后面都取最大就可以。

这个思想叫做前后缀分解

```C++
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int n=prices.size();
        vector<int> f(n+2);

        for(int i=1,minp=INT_MAX;i<=n;i++)
        {
            // 之所以是prices[i-1]是因为一个序号从0开始一个序号从1开始
            f[i]=max(f[i-1],prices[i-1]-minp); 
            
            // f[i]表示从1到i之间进行一次交易得到的最大收益
            minp=min(minp,prices[i-1]); // 更新minp
        }
        int res=0;

        for(int i=n,maxp=0;i;i--) // 从后往前遍历
        {
            // f[i-1]表示i之前的交易的收益最大值，maxp-prices[i-1]表示i之后交易的收益最大值
            res=max(res,f[i-1]+maxp-prices[i-1]);
            // 关于为什么都是i-1同一天，这是因为
            // 首先f[i-1]表示的不一定是在第i-1天卖出的交易的金额，而是从1～i-1的时间内交易收益的最大值
            // 其次，假如第一次交易是在x天买入，i-1天卖出，根据定义后一半部分的买入时间一定是i-1，卖出时间假设为y天，同在i-1天买入又卖出就相当于x天买入，y天卖出
            maxp=max(maxp,prices[i-1]); // 更新maxp
        }
        return res;
    }
};
```



 1004.最大连续1的个数3（经典双指针问题，类似于没有重复数字的子序列）（没有看代码独立完成了）

问题可以等价为，寻找一个最长的10区间，其中最多包含k个0

用i表示右端点，j表示当右端点固定为i之后能满足j-i中0的个数小于等于k的最靠左的端点

证明只需要走一遍（证明单调性）使用反证法

假设j-i是一个满足条件的区间，i往右边走1到i'，假设此时i'对应的左端点的j'在j的右边

由于j'-i'中0的个数小于等于k，那么j'-i中0的个数肯定也小于等于k，那么对应i的最左端点不应该是j而是j'，假设不成立。因此推出在i往右走的时候对应的j也会往右走（或者不变），反正不会往左走。因此证明有单调性，可以用双指针来完成。

![截屏2022-10-06 11.28.42.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-06%2011.28.42.png)

```C++
class Solution {
public:
    int longestOnes(vector<int>& nums, int k) {
        int res=0;
        int cnt=0;

        for(int i=0,j=0;i<nums.size();i++){
            if(nums[i]==0) cnt++;
            while(cnt>k){
                if(nums[j]==0) cnt--;
                j++;
            }
            res=max(res,i-j+1);
        } 
        return res;
    }
};
```



 207.课程表（拓扑排序模板题）-对应acwing848模板题 有向图的拓扑排序

有向图的拓扑排序问题

假如ab两门课，想学b必须先修a，那就一条边从a指向b。如果图有拓扑排序表示有解，如果不存在拓扑序表示无解（或者是判断是否有环，没有环才有拓扑排序）

算法步骤：

1.统计图中所有点的入度，从入度为0的点开始

2.将所有入度为0的点放进队列

3.宽搜框架：在所有入度为0的点中随便选择一个。把以这个点出发的点的入度-1，然后判断是否有入度变为0的点。如果有就也加入队列。

4.宽搜完判断是不是所有点都被遍历过，如果是就表示有拓扑排序

```C++
class Solution {
public:
    bool canFinish(int n, vector<vector<int>>& edges) {
        vector<vector<int>> g(n); // 邻接表
        vector<int> d(n); // 所有点的入度
        for(auto& e:edges){
            int a=e[0],b=e[1];
            g[a].push_back(b); // 插入一条a指向b的边
            d[b]++;
        }

        queue<int> q;
        for(int i=0;i<n;i++)
            if(d[i]==0) q.push(i);

        int cnt=0; // 统计遍历了几个点
        while(q.size()){
            auto t=q.front();
            q.pop();
            cnt++;
            for(auto i:g[t]) // 遍历以t点为出发点的边，将他们的入度-1然后判断是否入度变为0，如果为0则加入队列
            {
                d[i]--;
                if(d[i]==0) q.push(i);
                // 可以简化为一行
                // if(--d[i]==0) q.push(i);
            }
        }

        return cnt==n;

    }
};
```



重点复习：lc 33.搜索旋转排序数组 143.重排序链表



 84.柱状图中最大的矩形（单调栈的经典用法）

对于每个矩形i，找到其左右两边小于它的最远边界l[i]和r[i]

矩形的高度是h[i]，宽度是r[i]-l[i]-1

![截屏2022-10-06 14.37.42.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-06%2014.37.42.png)

```C++
class Solution {
public:
    int largestRectangleArea(vector<int>& h) {
        int n=h.size();
        vector<int> l(n);
        vector<int> r(n);
        stack<int> stk;
		
        // 使用两遍单调栈 分别找到数i在左边和右边的小于它的边界
      
        for(int i=0;i<n;i++){
            while(stk.size() && h[stk.top()]>=h[i]) stk.pop();
            if(stk.size()) l[i]=stk.top();
            else l[i]=-1; // 任何一个数都比它大，所以左边的边界可以取到-1
            stk.push(i);
        }

        stk=stack<int>(); // 将stack清空

        for(int i=n-1;i>=0;i--){
            r[i]=i+1;
            while(stk.size() && h[stk.top()]>=h[i]) stk.pop();
            if(stk.size()) r[i]=stk.top();
            else r[i]=n;
            stk.push(i);
        }

        int res=0;
        for(int i=0;i<n;i++){
            res=max(res,(r[i]-l[i]-1)*h[i]);
        }
        return res;
    }
};
```



 85.最大矩形

最直接就是枚举左上角和右下角的坐标，这样的复杂度是n的6次方

判断是否是全1的方式可以用增量方式，但是复杂度也是n的4次方

对于每一行来说，我们都可以看看以这一行元素为底的数字中向上最多有多少个连续的1（向上连续的1的高度或者长度是多少），可以看作柱状图的柱子

因此找最大的1构成的矩形的问题，可以看作在每一行上进行一次上一题中柱状图找最大矩形的问题。

![截屏2022-10-06 15.50.20.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-06%2015.50.20.png)

```C++
class Solution {
public:
    int largestRectangleArea(vector<int>& h) {
        int n=h.size();
        vector<int> l(n);
        vector<int> r(n);
        stack<int> stk;
        
        // 使用两遍单调栈 分别找到数i在左边和右边的小于它的边界
    
        for(int i=0;i<n;i++){
            while(stk.size() && h[stk.top()]>=h[i]) stk.pop();
            if(stk.size()) l[i]=stk.top();
            else l[i]=-1; // 任何一个数都比它大，所以左边的边界可以取到-1
            stk.push(i);
        }

        stk=stack<int>(); // 将stack清空

        for(int i=n-1;i>=0;i--){
            r[i]=i+1;
            while(stk.size() && h[stk.top()]>=h[i]) stk.pop();
            if(stk.size()) r[i]=stk.top();
            else r[i]=n;
            stk.push(i);
        }

        int res=0;
        for(int i=0;i<n;i++){
            res=max(res,(r[i]-l[i]-1)*h[i]);
        }
        return res;
    }

    int maximalRectangle(vector<vector<char>>& matrix) {
        if(matrix.empty() || matrix[0].empty()) return 0;
        int n=matrix.size(),m=matrix[0].size();
        vector<vector<int>> h(n,vector<int>(m));

        for(int i=0;i<n;i++){
            for(int j=0;j<m;j++){
                if(matrix[i][j]=='1'){
                    if(i>0) h[i][j]=1+h[i-1][j];
                    else h[i][j]=1;
                }
            }
        }

        int res=0;
        for(int i=0;i<n;i++){
            res=max(res,largestRectangleArea(h[i]));
        }

        return res;
    }
};
```



二分查找模板题（背一下每次mid是什么，以及对应的区间更新方式）



 200.岛屿数量（dfs模板题 flood-fill算法）

计算联通分量的个数，不需要恢复现场

```C++
class Solution {
public:
    int n,m;
    int dx[4]={1,0,-1,0},dy[4]={0,1,0,-1};

    int numIslands(vector<vector<char>>& grid) {
        if(grid.empty() || grid[0].empty()) return 0;
        n=grid.size(),m=grid[0].size();
        int res=0;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                if(grid[i][j]=='1')
                {
                    dfs(grid,i,j);
                    res++;
                }
        return res;
                    
    }

    void dfs(vector<vector<char>>& grid, int x, int y){
        grid[x][y]='0';
        for(int i=0;i<4;i++){
            int xx=x+dx[i],yy=y+dy[i];
            if(xx>=0 && xx<n && yy>=0 && yy<m && grid[xx][yy]=='1'){
                dfs(grid,xx,yy);
            }
        }
    }
};
```



 22.括号生成

n个左括号和右括号

合法括号序列的两个条件：

1. 任意前缀中左括号数量大于等于右括号数量

2. 左右括号数量相等

有多少个合法括号序列：卡特兰数

思路：用递归（dfs），每次判断一个位置上应该添加左括号还是右括号，输出所有合法的序列

如何判断：

3. 任何时候都可以添加左括号（前提是左括号数量不大于n）

4. 在前面的序列中，要左括号数量严格大于右括号数量时才能添加右括号（在相等的情况下，如果右括号+1就不满足上面的“任意前缀中左括号数量大于等于右括号数量”条件），且右括号的总数量不大于n

```C++
class Solution {
public:
    vector<string> res;

    vector<string> generateParenthesis(int n) {
        dfs(n,0,0,"");
        return res;
    }

    void dfs(int n, int lc, int rc, string seq){
        if(lc==n && rc==n) res.push_back(seq);

        if(lc<n){
            dfs(n,lc+1,rc,seq+'(');
        }
        if(rc<n && lc>rc){
            dfs(n,lc,rc+1,seq+')');
        }
    }
};
```



 322.零钱兑换

```C++
class Solution {
public:  
    int coinChange(vector<int>& coins, int amount) {
        // 完全背包问题
        // amount就是背包容量，coins就是物品的重量
        vector<int> f(amount+1,1e8);
        f[0]=0;

        for(int i=0;i<coins.size();i++){
            // 枚举每一个硬币的面额大小
            for(int j=coins[i];j<=amount;j++){ 
                // 更新大于等于这个面额的
                f[j]=min(f[j-coins[i]]+1,f[j]);
                // f[j-c]+1表示一个用当前coin[i]的硬币的一个个数和f[j-c]的个数相加的方案
            }
        }
        if(f[amount]==1e8) return -1;
        return f[amount];
    }
};
```



 129.求根节点到叶节点数字之和（独立完成）

```C++
class Solution {
public:
    vector<int> temp;
    int res=0;
    
    int sumNumbers(TreeNode* root) {
        if(root) dfs(root);
        return res;
    }

    void dfs(TreeNode* node){
        temp.push_back(node->val);
        if(!node->left && !node->right){
            int sum=0;
            for(auto s:temp) sum=sum*10+s;
            res+=sum;
        }
        if(node->left) dfs(node->left);
        if(node->right) dfs(node->right);
        temp.pop_back();
    }
};
```



 662.二叉树最大宽度

宽度优先搜索来层层遍历树上的节点，对于每一层都重新从1开始编号，不然很快编号就会爆int

![截屏2022-10-07 09.59.01.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-07%2009.59.01.png)

```C++
class Solution {
public:
    int widthOfBinaryTree(TreeNode* root) {
        if(!root) return 0;
        queue<pair<TreeNode*,long>> q; // first表示这个节点的指针，second表示这个节点的编号（可能会爆int所以用long）
        q.push({root,1});
        int res=1;

        while(q.size()){ // BFS来遍历树中的所有节点
            int sz=q.size();
            int l=q.front().second;
            int r;

            for(int i=0;i<sz;i++){
                auto t=q.front(); 
                q.pop(); // 取得队头元素然后pop
                auto v=t.first;
                auto p=t.second-l+1; // 将这个编号更新为从1开始的编号（不然非常快就爆int了）
                r=t.second; // 原本的编号（还没有更新为从1开始的编号）
                if(v->left) q.push({v->left,(long)p*2});
                if(v->right) q.push({v->right,(long)p*2+1});
            }
            res=max(res,r-l+1); // 更新一下最大“宽度”
        }
        return res;

    }
};
```



复习：lc 98.验证二叉搜索树



 1.两数之和

```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> s;
        int n=nums.size();
        vector<int> res({0,0});
        for(int i=0;i<n;i++){
            int r=target-nums[i];
            if(s.count(r)){
                res[0]=i;
                res[1]=s[r];
            }
            s[nums[i]]=i;
        }
        return res;
    }
};
```



 1800.最大升序子数组和

```C++
class Solution {
public:
    int maxAscendingSum(vector<int>& nums) {
        int n=nums.size();
        int res=0;
        
        for(int i=0;i<n;i++){
            int x=i; // 左指针，往左边走
            int t=0;
            // 如果下一个数比当前数更小，就往左走
            while(x && nums[x]>nums[x-1]) t+=nums[x],x--;
            if(x>=0) t+=nums[x];
            res=max(res,t);
        }
        return res;
    }
};
```



 6.Z字形变换

```C++
class Solution {
public:
    string convert(string s, int n) {
        string res;
        if(n==1) return s;
        for(int i=0;i<n;i++){
            if(i==0 || i==n-1){
                for(int j=i;j<s.size();j+=2*n-2){
                    res+=s[j];
                }
            } else{
                for(int j=i,k=2*n-2-i;j<s.size() || k<s.size(); j+=2*n-2,k+=2*n-2){
                    if(j<s.size()) res+=s[j];
                    if(k<s.size()) res+=s[k];
                }
            }
        }
        return res;
    }
};
```



 7.整数反转

```C++
class Solution {
public:
    int reverse(int x) {
        long long r=0;
        while(x){
            r=r*10+x%10; // 每次更新一下
            x/=10; // 
        }
        if(r>INT_MAX) r=0;
        if(r<INT_MIN) r=0;
        return r;

    }
};
```



 11.盛最多水的容器

双指针来做，维护左右两个条条，每次判断一下左右两个条哪个高度较低就移动哪个（为什么一定是正确的呢？）

![截屏2022-10-07 14.36.43.png](/Users/caizy/Desktop/good job/pics/%E6%88%AA%E5%B1%8F2022-10-07%2014.36.43.png)

假设中间有一个最优解，左右两个指针肯定会在慢慢移动的过程中到达两个最优解指针

假设左边先到最优解左指针，此时右指针还需要往前（左）移动一定距离才会到

此时的右指针j对应h[j]肯定是小于等于左（最优解）指针的h[i]

反证：假设h[j]是大于h[i]的（如图所示），那么此时i和j围成的面积肯定大于i和j左边的右最优解指针围成的最优解面积，那么最优解面积就不会是最优解，所以假设不成立，此时的h[j]一定小于等于h[i]

所以在h[i]>h[j]的时候就移动右边指针，j--

```C++
class Solution {
public:
    int maxArea(vector<int>& h) {
        int n=h.size();
        int res=0;

        for(int i=0,j=n-1;i<j;)
        {
            res=max(res,min(h[j],h[i])*(j-i)); // 更新一下盛水容器的体积
            if(h[i]<h[j]) i++; // 这里可以用反证法证明
            else j--;
        }
            
        return res;   
    }
};
```



 206.反转链表

```C++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next) return head;
      
        int k=0; // 计算出链表的长度，后面只需要翻转k-1次
        auto a=head;
        while(a){
            a=a->next;
            k++;
        }
        
        a=head;
        auto b=a->next;

        for(int i=0;i<k-1;i++){
            // 依次翻转两个节点之间的指针
            auto c=b->next;
            b->next=a;
            a=b;
            b=c;
        }
        // 非常重要！！尾节点现在是head了，要将尾节点的next置为NULL
        head->next=NULL;
        return a;
    }
};
```



 215.数组中的第K个最大元素（快速选择算法）

快选算法就是快速排序每次只递归一边

5. 每次选择一个x数，小于x的放左边，大于x的放右边

6. 如果第K大元素在左边就去左边递归

7. 如果第K大元素在右边就去右边递归

```C++
class Solution {
public:
    int quick_sort(vector<int> &nums, int l, int r, int k){
        if(l==r) // 说明区间里就一个数
            return nums[k];
        int x=nums[l],i=l-1,j=r+1;
        while(i<j){
            do i++;while(nums[i]>x);
            do j--;while(nums[j]<x);
            if(i<j) swap(nums[i],nums[j]);
        }
        if(k<=j) // 递归左边
                return quick_sort(nums,l,j,k);
        else return quick_sort(nums,j+1,r,k);
    }
    int findKthLargest(vector<int>& nums, int k) {
        return quick_sort(nums,0,nums.size()-1,k-1);
    }
};
```



##### 3 无重复数字的最长子串 *

```C++
class Solution {
public:
//滑动窗口
    int lengthOfLongestSubstring(string s) {
        unordered_map<char,int> count; // 字符和出现次数的对应
        
        int res=0;
        for(int i=0,j=0;i<s.size();i++)
        {
            count[s[i]]++; // 刚加入的字符s[i]的个数++
            while(count[s[i]]>1)
                count[s[j++]]--; // 左指针往右移 直到移动到没有重复字符
            res=max(res,i-j+1); // 更新最长长度
        }
        return res;
    }
};
```

##### 160 相交链表 思路巧妙

两个指针分别从A和B链表的表头开始往后走，走到尾的时候去另一个链表的表头继续走。

走到两个链表指针相等的时候，p要么是两个链表相交的点，要么是null（表示不相交）

```C++
class Solution {
public:
    ListNode *getIntersectionNode(ListNode *headA, ListNode *headB) {
        auto p=headA,q=headB;
        while(p!=q)
        {
            p=p?p->next:headB;
            q=q?q->next:headA;
        }
        return p;
    }
};
```



##### 206 反转链表 *

迭代或者递归地反转链表

**迭代**：比较常见比较简单

```C++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head) return NULL;
        auto a=head,b=a->next;
        while(b)
        {
            auto c=b->next;
            b->next=a;
            a=b;
            b=c;
        }
        head->next=NULL;
        return a;
    }
};
```

**递归**：

```C++
class Solution {
public:
    ListNode* reverseList(ListNode* head) {
        if(!head || !head->next) return head;
        auto tail=reverseList(head->next);
        head->next->next=head;
        head->next=NULL;
        return tail;
    }
};
```



##### 92 反转链表2 难

把某一段链表反转(m到n之间的部分反转)

```C++
class Solution {
public:
    ListNode* reverseBetween(ListNode* head, int m, int n) {
        auto dummy=new ListNode(-1);
        dummy->next=head;
​
        auto a=dummy;
        for(int i=0;i<m-1;i++) a=a->next;
​
        auto b=a->next,c=b->next;
        // 反转b和c之间的链表
        for(int i=0;i<n-m;i++)
        {
            auto t=c->next;
            c->next=b;
            b=c,c=t;
        }
        // 这里的顺序绝对不能改
        a->next->next=c;
        a->next=b;
        return dummy->next;
    }
};
```



##### 169 多数元素 *

找到一个数组中出现次数大于[n/2]（下取整）的元素

思路离奇，背过就好

```C++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        int r,c=0;
        for(auto x:nums)
        {
            if(!c) r=x,c=1;
            else if(r==x) c++;
            else  c--;
        }
        return r;
    }
};
```



##### 1 两数之和 *

```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int,int> pos;
        
        for(int i=0;i<nums.size();i++)
        {
            int r=target-nums[i];
            if(pos.count(r)) // r是否出现过
                return {pos[r],i};
            pos[nums[i]]=i; // nums[i]与位置i对应
        }
        return {};
        
    }
};
```



##### 20 有效的括号 *

```C++
class Solution {
public:
    bool isValid(string s) {
        stack<char> stk;
​
        for(char c:s)
        {
            if(c=='[' || c=='{' || c=='(') stk.push(c);
            else{
                // 注意这里的abs
                if(stk.size() && abs(c-stk.top())<=2) stk.pop();
                else return false;
            }
        }
        return stk.empty();
    }
};
```



##### 145 字符串相加 *

高精度相加（两数相加）的模板要背过

```C++
class Solution {
public:
    // 两数相加的模板
    vector<int> add(vector<int>& A, vector<int>& B)
    {
        vector<int> C;
        for(int i=0,t=0;i<A.size() || i<B.size() || t;i++)
        {
            if(i<A.size()) t+=A[i];
            if(i<B.size()) t+=B[i];
            C.push_back(t%10);
            t/=10;
        }
        return C;
    }
    string addStrings(string num1, string num2) {
        vector<int> A,B;
        for(int i=num1.size()-1;i>=0;i--) A.push_back(num1[i]-'0');
        for(int i=num2.size()-1;i>=0;i--) B.push_back(num2[i]-'0');
        auto C=add(A,B);
        string c;
        for(int i=C.size()-1;i>=0;i--) c+=to_string(C[i]);
        return c;
    }
};
```



##### 239 滑动窗口最大值 hard（打印下来）

使用单调队列，一端删除，另一端插入也删除

```C++
class Solution {
public:
    vector<int> maxSlidingWindow(vector<int>& nums, int k) {
        deque<int> q; // 双端队列 队尾插 队头删
        vector<int> res;
        for(int i=0;i<nums.size();i++){
            // 左边界是i-k+1 如果比队头大 就让队头出队列
            if(q.size() && i-k+1 > q.front()) q.pop_front();
            // 首先删除队尾元素
            while(q.size() && nums[i]>=nums[q.back()]) q.pop_back();
            // 加入当前元素
            q.push_back(i);
            // 把当前最大值加入res
            if(i>=k-1) res.push_back(nums[q.front()]);
        }
        return res;
    }
};
```



##### 179 最大数 新的比较方式排序 *

代码很短但是思路很难

定义了一种新的比较方式

```C++
class Solution{
    public:
    string largestNumber(vector<int>& nums){
        sort(nums.begin(),nums.end(),[](int x, int y){
            string a=to_string(x),b=to_string(y);
            return a+b>b+a;
        });
        
        string res;
        for(auto x:nums) res+=to_string(x);
        int k=0;
        while(k+1<res.size() && res[k]=='0') k++;
        return res.substr(k);
    }
};
```



##### 121 买卖股票的最佳时机 *

```Plain Text
class Solution {
public:
    int maxProfit(vector<int>& prices) {
        int res=0;
        for(int i=0,minp=INT_MAX;i<prices.size();i++){
            res=max(res,prices[i]-minp);
            minp=min(minp,prices[i]); // minp记录的是从i位置往前的最小值
        }
        return res;
    }
};
```



##### 83 删除排序链表中的重复元素

每个相同元素留下第一个

```Plain Text
class Solution {
public:
    ListNode* deleteDuplicates(ListNode* head) {
        if(!head) return head;
​
        auto cur=head;
        for(auto p=head->next;p;p=p->next)
            if(p->val!=cur->val) // 相邻元素不相等就表示没有重复
                cur=cur->next=p;
        
        cur->next=NULL;
        return head;
    }
};
```



##### 78 子集 打印下来

可以用递归或者迭代来写（二进制的形式）

```C++
class Solution {
public:
    vector<int> path;
    vector<vector<int>> subsets(vector<int>& nums) {
        vector<vector<int>> res;
        int n=nums.size();
        for(int i=0;i<1<<n;i++){
            vector<int> path;
            for(int j=0;j<n;j++){
                if(i>>j & 1) path.push_back(nums[j]); // 位移到最后一位
            res.push_back(path);
            }
        }
        return res;
    }
};
```



##### 94 二叉树中序遍历

迭代方式写

```C++
class Solution {
public:
    vector<int> inorderTraversal(TreeNode* root) {
        vector<int> res;
        stack<TreeNode*> stk;
    // 注意第一个是while 第二个是if
        while(root || stk.size())
        {
            while(root)
            {
                stk.push(root);
                root=root->left;
            }
            if(stk.size())
            {
                root=stk.top();
                stk.pop();
                res.push_back(root->val);
                root=root->right;
            }
        }
        return res;
    }
};
```



##### 15 三数之和 暂时不熟

找到和为0的三个数。可以用双指针算法，枚举其中一个数，另外两个数就用双指针算法。

```C++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        vector<vector<int>> res;
        sort(nums.begin(),nums.end());
    // 枚举指针 第一个直接循环 后两个指针用滑动窗口
        for(int i=0;i<nums.size();i++){
            // 有重复的数
            if(i>0 && nums[i]==nums[i-1]) continue;
            for(int j=i+1,k=nums.size()-1;j<k;j++){
                // 有重复的数
                if(j>i+1 && nums[j]==nums[j-1]) continue;
                // 注意这里是k-1
                while(j<k-1 && nums[i]+nums[j]+nums[k-1]>=0) k--;
                if(nums[i]+nums[j]+nums[k]==0){
                    res.push_back({nums[i],nums[j],nums[k]});
                }
            }
        }
        return res;
    }
};
```



##### 141 环形链表

快慢指针（包含在双指针算法内）

初始化的时候快指针比慢指针快1步。然后快指针走两步，慢指针走一步。

快慢指针方法1:

```C++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        if(!head || !head->next) return false;
        auto s=head, f=head->next;
        // 快指针f比慢指针s快一步
        
        while(f)
        {
            s=s->next,f=f->next;
            if(!f) return false;
            else f=f->next;
            if(s==f) return true;
        }
        return false;
    }
};
```

快慢指针方法2:

```C++
class Solution {
public:
    bool hasCycle(ListNode *head) {
        auto* f=head,s=head;
        while(f && f->next)
        {
            f=f->next->next,s=s->next;
            if(f==s)//之所以放在里面是因为至少2个结点才能成环
                return true;
        }
        return false;
    }
};
```



##### 剑指offer 61 扑克牌中顺子

判断5张牌是否是顺子，0表示大小王，可以代替任何数

```C++
class Solution{
    public:
    bool isContinuous(vector<int> nums){
        if(nums.empty()) return false;
        
        sort(nums.begin(),nums.end());
        int k=0;
        while(!nums[k]) k++;
        
        for(int i=k+1;i<nums.size();i++)
            if(nums[i]==nums[i-1]) return false;
        
        return (nums.back()-nums[k])<=4; // 最大和最小相差小于等于4就可以用0补全
    }
};
```



##### 718 最长重复子数组

如果是最长子序列（可以不连续）就用dp来做

这个题可以用字符串哈希来做。

通过O(n)的预处理，可以O(1)求出任意区间的值

字符串哈希来判断某两端是否相等。就是r-k算法（是啥）

```C++
// 字符串哈希的方法 有bug

typedef unsigned long long ULL;
const int P=131;

class Solution {
public:
    int n,m;
    vector<ULL> ha,hb,p;

    ULL get(vector<ULL>& h, int l, int r){
        return h[r]-h[l-1]*p[r-l+1];
    }

    bool check(int mid){
        unordered_set<ULL> hash;
        for(int i=mid;i<=n;i++) hash.insert(get(ha,i-mid+1,i));
        for(int i=mid;i<=m;i++)
            if(hash.count(get(hb,i-mid+1,i)))
                return true;
        return false;
    }

    int findLength(vector<int>& A, vector<int>& B) {
        n=A.size(),m=B.size();
        ha.resize(n+1),hb.resize(m+1),p.resize(n+1);
        for(int i=1;i<=n;i++) ha[i]=ha[i-1]+P+A[i-1];
        for(int i=1;i<=m;i++) hb[i]=hb[i-1]+P+B[i-1];
        p[0]=1;
        for(int i=1;i<=n;i++) p[i]=p[i-1]*P;

        int l=0,r=n;
        while(l<r){
            int mid=(l+r+1)>>1;
            if(check(mid)) l=mid;
            else r=mid-1;
        }
        return r;

    }
};
```

**简单的dp做法**

```C++
class Solution {
public:
    int findLength(vector<int>& A, vector<int>& B) {
        vector<vector<int>> f(A.size()+1,vector<int>(B.size()+1,0));
        int res=0;

        for(int i=1;i<=A.size();i++){
            for(int j=1;j<=B.size();j++){
                if(A[i-1]==B[j-1]) f[i][j]=f[i-1][j-1]+1; // 由于一定是连续的 这里就只用和f[i-1][j-1]来判断
                res=max(res,f[i][j]);
            }
        }
        return res;
    }
};
```



##### 64 最小路径和 *

DP法 分为往下走和往右走两个部分

```C++
class Solution {
public:
    // dp问题
    int minPathSum(vector<vector<int>>& grid) {
        int n=grid.size();
        if(!n) return 0;
        int m=grid[0].size();

        vector<vector<int>> f(n,vector<int>(m,INT_MAX));
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
            {
                if(!i && !j) f[i][j]=grid[i][j]; // 起点
                else{
                    if(i) f[i][j]=min(f[i][j],f[i-1][j]+grid[i][j]); // 往右走
                    if(j) f[i][j]=min(f[i][j],f[i][j-1]+grid[i][j]); // 往下走
                }
            }      
        return f[n-1][m-1];
    }
};
```



##### 62 不同路径 *

非常简单的DP问题

```C++
// dp方法
class Solution {
public:
    int uniquePaths(int m, int n) {
        vector<vector<int>> f(n,vector<int>(m));
        f[0][0]=1;
        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
            {
                if(i) f[i][j]+=f[i-1][j];
                if(j) f[i][j]+=f[i][j-1];
            }
        return f[n-1][m-1];
    }
};
```

组合数方法：一共要往下走n-1步，往右走m-1步，所以可以转化为计算组合数（不详细展开）



##### 63 不同路径2 *

有的格子有障碍，不能走到障碍物上。如果某个格子没有障碍，就按照上面的公式计算：

`if(i) f[i][j]+=f[i-1][j]`

`if(j) f[i][j]+=f[i][j-1]`

否则就`f[i][j]=0`表示不能走

```Plain Text
class Solution {
public:
    int uniquePathsWithObstacles(vector<vector<int>>& grid) {
        int n=grid.size();
        if(!n) return 0;
        int m=grid[0].size();

        vector<vector<int>> f(n,vector<int> (m));

        for(int i=0;i<n;i++)
            for(int j=0;j<m;j++)
                if(!grid[i][j])
                {
                    // 第一个格子
                    if(!i && !j) f[i][j]=1;
                    else{
                        if(i) f[i][j]+=f[i-1][j];
                        if(j) f[i][j]+=f[i][j-1];
                    }
                }
            // 这个格子不能走
                else f[i][j]=0;
        
        return f[n-1][m-1];
    }
};
```



##### 70 爬楼梯 * 要记清楚

就是最简单dp问题/斐波那契数列

```C++
class Solution {
public:
    int climbStairs(int n) {
        int res;
        vector<int> f(n+1);

        for(int i=0;i<=n;i++)
        {
            if(!i) f[i]=1;
            if(i) f[i]+=f[i-1];
            if(i>1) f[i]+=f[i-2];
        }

        return f[n];
    }
};
```

fib数列法

```Plain Text
class Solution {
public:
    int climbStairs(int n) {
        int a=1,b=1;

        while(--n)
        {
            int c=a+b;
            a=b,b=c;
        }
        return b;
    }
};
```



##### 110 平衡二叉树 *

需要借助dfs来计算二叉树的高度

```C++
class Solution {
public:
    bool res;
    
    bool isBalanced(TreeNode* root) {
        res=true;
        dfs(root);
        return res;
    }

    int dfs(TreeNode* root){
        if(!root) return 0;
        int lh=dfs(root->left);
        int rh=dfs(root->right);
        if(abs(lh-rh)>1) res=false;
        return max(lh,rh)+1;
    }
};
```



##### 39 组合总和 *

每个数可以选择无限个，凑出这个target

用dfs暴搜

```C++
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> combinationSum(vector<int>& c, int target) {
        dfs(c,0,target);
        return ans;
    }
  
    // c是数组 u是目前在的第几个数 target是目前要凑到的总和
    void dfs(vector<int>& c, int u, int target)
    {
        if(target==0) // 方案凑出来了
        {
            ans.push_back(path);
            return;
        }
        if(u==c.size()) return;

        for(int i=0;c[u]*i<=target;i++)
        {
            dfs(c,u+1,target-c[u]*i); // 选择i个c[u]的情况
            path.push_back(c[u]);
        }

        // 恢复现场
        for(int i=0;c[u]*i<=target;i++)
            path.pop_back();
    }
};
```



##### 40 组合总数2 *

每个数只有有限个（类似于组合背包问题）

在循环过程中加一个限制即可。

可以用排序或者哈希表的方法来求出每个数的个数。

```C++
class Solution {
public:
    vector<vector<int>> ans;
    vector<int> path;

    vector<vector<int>> combinationSum2(vector<int>& c, int target) {
        sort(c.begin(),c.end());
        // 排序之后可以方便的得到每个数的个数
        dfs(c,0,target);

        return ans;
    }

    void dfs(vector<int>& c, int u, int target)
    {
        if(target==0)
        {
            ans.push_back(path);
            return;
        }
        if(u==c.size()) return;

        int k=u+1;
        while(k<c.size() && c[k]==c[u]) k++; // 统计c[u]的个数
        int cnt=k-u; // cnt记录c[u]的可使用个数

        for(int i=0;c[u]*i<=target && i<=cnt;i++)
        {
            dfs(c,k,target-c[u]*i);
            path.push_back(c[u]);
        }

        for(int i=0;c[u]*i<=target && i<=cnt;i++)
            path.pop_back();
    }
};
```



##### 56 合并区间 *

这是一个模板题

```C++
class Solution {
public:
    vector<vector<int>> merge(vector<vector<int>>& a) {
        vector<vector<int>> res;
        if(a.empty()) return res;

        sort(a.begin(),a.end()); // 根据左端点排序
        int l=a[0][0],r=a[0][1]; // 第一个区间的左右端点
        // 从第二个区间开始枚举
        for(int i=1;i<a.size();i++)
        {
            if(a[i][0]>r){
                // 当前区间可以保存
                res.push_back({l,r});
                // 更新为当前这个区间
                l=a[i][0],r=a[i][1];
            }
            // 合并区间
            else r=max(r,a[i][1]);
        }
        res.push_back({l,r});
        return res;
    }
};
```



##### 剑指offer21 调整数组顺序使奇数位于偶数前面 类似快速排序

```Plain Text
class Solution {
public:
    vector<int> exchange(vector<int>& nums) {
        int i=-1,j=nums.size();
        while(i<j){
            do i++;while(i<nums.size() && nums[i]%2==1);
            do j--;while(j>=0 && nums[j]%2==0);
            if(i<j) swap(nums[i],nums[j]);
        }
        return nums;
    }
};
```



##### 912 快速排序

```Plain Text
class Solution {
public:
    vector<int> sortArray(vector<int>& nums) {
        quick_sort(nums,0,nums.size()-1);
        return nums;
    }

    void quick_sort(vector<int>& nums, int l, int r){
        if(l>=r) return;
    // 注意这里要写l+r>>1 不然会超时
        int x=nums[l+r>>1],i=l-1,j=r+1;
        while(i<j){
            do i++;while(nums[i]<x);
            do j--;while(nums[j]>x);
            if(i<j) swap(nums[i],nums[j]);
        }
        quick_sort(nums,l,j);
        quick_sort(nums,j+1,r);
    }
};
```



##### 215 数组中第K个最大元素 *

利用快速排序模板

```Plain Text
class Solution {
public:
    int quick_sort(vector<int>& nums, int l, int r, int k)
    {
        if(l==r) return nums[k];
        int x=nums[l],i=l-1,j=r+1;
        
        while(i<j)
        {
            do i++;while(nums[i]>x);
            do j--;while(nums[j]<x);
            if(i<j) swap(nums[i],nums[j]);
        }
        if(k<=j) return quick_sort(nums,l,j,k);
        else return quick_sort(nums,j+1,r,k);
    }

    int findKthLargest(vector<int>& nums, int k) {
        return quick_sort(nums,0,nums.size()-1,k-1);
    }
};
```



##### 704 二分查找 *

默写二分查找模板就可以了

```Plain Text
class Solution {
public:
    int search(vector<int>& nums, int target) {
        int l=0,r=nums.size()-1;
        while(l<r)
        {
            int mid=l+r>>1; // 取重点判断左还是右
            if(nums[mid]>=target) r=mid;
            else l=mid+1;
        }
        // 注意下面都是r或者l 不能写mid
        if(nums[r]!=target) return -1;
        return r;
    }
};
```



##### 26 删除排序数组中的重复项 *

```Plain Text
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        int k=0;
        for(int i=0;i<nums.size();i++)
            if(i==0 || nums[i]!=nums[i-1]) // 重复元素肯定与上一个元素相等
                nums[k++]=nums[i];
        return k;
    }
};
```



##### 88 合并两个有序数组 *

```Plain Text
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        int k=m+n-1;
        int i=m-1,j=n-1;

        while(i>=0 && j>=0)
        {
            if(nums1[i]>nums2[j]) nums1[k--]=nums1[i--];
            else nums1[k--]=nums2[j--];
        }

        while(j>=0) nums1[k--]=nums2[j--];
    }
};
```

##### 124 二叉树中最大路径和 hard

对于树中的路径，一般枚举树的最高点 本质上是树形dp

```Plain Text
class Solution {
public:
    int ans;

    int maxPathSum(TreeNode* root) {
        ans=INT_MIN;
        dfs(root);
        return ans;
    }

    int dfs(TreeNode* u){
        if(u==NULL) return 0;
        int left=max(0,dfs(u->left)),right=max(0,dfs(u->right));
        ans=max(ans,u->val+left+right);
        return u->val+max(left,right);
    }
};
```



##### 76 最小覆盖子串 hard

滑动窗口算法（双指针算法）的应用。能用双指针算法**一定要有单调性**。

类似于30题，建议去学会30题再来看

对于每个right都去求最近的left，使left到right这段包含有所有子串的字符

用两个哈希表，计算在整个字符串中的次数和在窗口内出现的次数

```C++
class Solution {
public:
    string minWindow(string s, string t) {
        unordered_map<char,int> hs,ht;
        for(auto c:t) ht[c]++; // t串当中每个字符的出现次数
        string res;
        for(int i=0,j=0;i<s.size();i++)
        {
            hs[s[i]]++;
            if(hs[s[i]]<=ht[s[i]]) cnt++; // 包含一个新的字符
​
            while(hs[s[j]]>ht[s[j]]) hs[s[j++]]--; // 有多余字符 j往前走
            if(cnt==t.size()) // 每个字符都包含了
            {
                if(res.empty() || i-j+1<res.size())
                    res=s.substr(j,i-j+1);
            }
        }
        return res;
    }
};
```



##### 53 最大子序和

```C++
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int res=INT_MIN;
        for(int i=0,last=0;i<nums.size();i++){
            last=nums[i]+max(last,0);
            res=max(res,last);
        }
        return res;
    }
};
```



##### 19 删除链表的倒数第N个节点 *

```C++
class Solution {
public:
    ListNode* removeNthFromEnd(ListNode* head, int k)
    {
        auto dummy=new ListNode(-1);
        dummy->next=head; //添加头节点dummy

        int n=0; // 计算出长度
        for(auto p=dummy;p;p=p->next) n++;

        auto p=dummy;

        for(int i=0;i<n-k-1;i++) p=p->next; //找到倒数第k+1个节点
        p->next=p->next->next; //删除倒数k个节点

        return dummy->next; //返回真正的头节点
    }
};
```



##### 543 二叉树的直径 *

dfs函数计算深度，每次更新一下maxd`max(lh+rh,maxd)`

```C++
class Solution {
public:
    int maxd;

    int diameterOfBinaryTree(TreeNode* root) {
        dfs(root);
        return maxd;
    }

    int dfs(TreeNode* root)
    {
        if(!root) return 0;
        int lh=dfs(root->left);
        int rh=dfs(root->right);
        maxd=max(maxd,lh+rh);
        return max(lh,rh)+1;
    }
};
```



##### 240 搜索二维矩阵 *

每行都是左小右大，每列都是上小下大。每次比较最右上角的数和当前数的关系。

```C++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        // 判断一下行和列是否为空
        if(matrix.empty() || matrix[0].empty()) return false;
        int n=matrix.size(),m=matrix[0].size();
        // 从二维矩阵的最右上角开始
        int i=0,j=m-1;
        while(i<n && j>=0)
        {
            int t=matrix[i][j];
            if(t==target) return true;
            // 去掉大于t的一列
            else if(t>target) j--;
            // 去掉小于t的一行
            else i++;
        }
        return false;
    }
};
```



##### 162 寻找峰值 *

寻找任何一个峰值。就使用二分算法就能得到答案。

```C++
class Solution {
public:
    int findPeakElement(vector<int>& nums) {
        int l=0,r=nums.size()-1;
        while(l<r)
        {
            int mid=l+r>>1;
            // 注意这一步
            if(nums[mid]>nums[mid+1]) r=mid;
            else l=mid+1;
        }
        return r;
    }
};
```



##### 125 验证回文串 *

只考虑字母和数字，不考虑其他（指针走的时候跨过不是字母和数字的）

```C++
class Solution {
public:
​
    bool check(char c){
        return c>='a' && c<='z' || c>='A' && c<='Z' || c>='0' && c<='9';
    }
    bool isPalindrome(string s) {
        for(int i=0,j=s.size()-1;i<j;i++,j--)
        {
            while(i<j && !check(s[i])) i++;
            while(i<j && !check(s[j])) j--;
            if(i<j && tolower(s[i])!=tolower(s[j])) return false;
        }
        return true;
    }
};
```



##### 268 缺失数字 *

```C++
class Solution {
public:
    int missingNumber(vector<int>& nums) {
        int n=nums.size();
        int res=n*(n+1)/2;
        for(auto x:nums) res-=x;
        return res;
    }
};
```



##### 198 打家劫舍

简单的dp问题

```C++
class Solution {
public:
    int rob(vector<int>& nums) {
        int n=nums.size();
        vector<int> f(n+1),g(n+1);
        // f[i]表示1-i中选，选i的情况
        // g[i]表示1-i中选，不选i的情况

        for(int i=1;i<=n;i++)
        {
            // f[i] 必然不选i-1位
            f[i]=g[i-1]+nums[i-1];
            // g[i] 可以选i-1也可以不选i-1
            g[i]=max(f[i-1],g[i-1]);
        }
        return max(f[n],g[n]);
    }
};
```



##### 242 有效的字母异位词 *

判断两个字符串是不是字母异位词（就是相同的一堆字母，但是排列不一样）

就用哈希表来记录每个字母的数量，最后比较两个哈希表是否相等

```C++
class Solution {
public:
    bool isAnagram(string s, string t) {
        unordered_map<char,int> a,b;
        for(auto c:s) a[c]++;
        for(auto c:t) b[c]++;
        return a==b;
    }
};
```



##### 151 翻转字符串里的单词 打印下来

可能会有多余的空格等符号 删掉前后多余空格 调转单词的顺序

最好使用O(1)的时间复杂度

```C++
class Solution {
public:
    string reverseWords(string s) {
        int k=0;
        for(int i=0;i<s.size();i++){
            if(s[i]==' ') continue;
            int j=i,t=k; // 到了不是空格的位置
            while(j<s.size() && s[j]!=' ') s[t++]=s[j++]; // 一直走到单词结尾
        reverse(s.begin()+k,s.begin()+t); // 把单词翻转过来
        // 单词后加一个空格
        s[t++]=' ';
        // 更新指针
        k=t,i=j;      
        }
        // 删除最后的空格
        if(k) k--;
        // 删除从k到最后的空格
        s.erase(s.begin()+k,s.end());
        // 全部翻转
        reverse(s.begin(),s.end());
        return s;
    }
};
```



##### 103 二叉树的锯齿形层次遍历 *

```C++
class Solution {
public:
    vector<vector<int>> zigzagLevelOrder(TreeNode* root) {
        vector<vector<int>> res;
        if(!root) return res;
        int cnt=0;
        queue<TreeNode *> q;
        q.push(root);

        while(q.size())
        {
            vector<int> temp;
            int size=q.size();
            for(int i=0;i<size;i++)
            {
                auto node=q.front();
                temp.push_back(node->val);
                if(node->left) q.push(node->left);
                if(node->right) q.push(node->right);
                q.pop();
            }

            if(cnt%2)
                reverse(temp.begin(),temp.end());
            res.push_back(temp);
            cnt++;
        }
        return res;
    }
};
```



##### 剑指offer 10-2 青蛙跳台阶问题

就是斐波那契数列问题

```C++
class Solution {
public:
    int numWays(int n) {
        if(n==0) return 1;
        int fib1=1,fib2=1;
        int mod=1e9+7;
        while(--n){
            int c=(fib1+fib2)%mod;
            fib1=fib2;
            fib2=c;
        }
        return fib2;
    }
};
```



##### 470 用rand7实现rand10

```C++
class Solution {
public:
    int rand10() {
        int t=(rand7()-1)*7+rand7(); // 1～49的数字
        if(t>40) return rand10(); // 如果在41～49 回炉重造
        return (t-1)%10+1; // 40转化为1-10
    }
};
```



##### 93 复原IP地址 *

一个只有数字的字符串，分割成为一个合法的IP地址。

一个数一个数搜，确保每个数都在0——255之间，且每个数都不能有前导0。

给了n-1个放点的位置，放3个点。

```C++
class Solution {
public:
    vector<string> ans;

    vector<string> restoreIpAddresses(string s) {
        dfs(s,0,0,"");
        return ans;
    }

    void dfs(string& s, int u, int k, string path){
        if(u==s.size()){
            if(k==4){
                path.pop_back();
                ans.push_back(path);
            }
            return;
        }
        if(k==4) return; // 已经有4个数但是没有选完
        for(int i=u,t=0;i<s.size();i++){
            if(i>u && s[u]=='0') break; // 有前导0
            t=t*10+s[i]-'0';
            if(t<=255) dfs(s,i+1,k+1,path+to_string(t)+'.');
            else break; // 大于255
        }
    }
};
```



##### 283 移动零 *

```C++
class Solution {
public:
    void moveZeroes(vector<int>& nums) {
        if(nums.empty()) return;
        int i=0,j=0;
        // 把所有非0的都移动到前面去
        while(i<nums.size())
        {
            if(!nums[i]) i++;
            else nums[j++]=nums[i++];
        }
        // 把后面空缺的部分都设置为0
        while(j<nums.size())
            nums[j++]=0;
    }
};
```



##### 对称的二叉树

```C++
class Solution {
public:
    bool isSymmetric(TreeNode* root) {
        if(root==NULL) return true;
        return dfs(root->left,root->right);
    }
    bool dfs(TreeNode* p, TreeNode* q){
        if(!p && !q) return true;
        if(!p || !q || p->val!=q->val) return false;
        return dfs(p->left,q->right)&&dfs(p->right,q->left);
    }
};
```

