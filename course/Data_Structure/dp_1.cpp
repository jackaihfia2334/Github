#include <iostream>
#define MAX 10
int m,n;  //������к���
int arr[MAX][MAX]; //����ԭʼ����Ԫ��
int F[MAX][MAX]; //�����ռ��������Ӳ����
int H[MAX][MAX];//���л��ݲ���ʱ������·������

void HuiSu();  //��������
void FindMax();
int Max(int a,int b);

int main()
{
	printf("�����뷽���������������(�ո����)\n");    //������������±�0������
	scanf("%d%d",&m,&n);
    for(int i=0;i<=m;i++) //��0��ȫ����0
    {arr[i][0]=0;}
    for(int j=0;j<=n;j++)//��0��ȫ����0
    {arr[0][j]=0;}
	for(int i=1;i<=m;i++)
	{
		printf("�������%d�е�%d������(ֻ�ܳ���0��1)\n",i,n);
		for(int j=1;j<=n;j++)
		{
			scanf("%d",&arr[i][j]);
		}
	}
    printf("������%d��%d�е�ľ�����£�\n ",m,n);
    for(int i=1;i<=m;i++)
    {
        for(int j=1;j<=n;j++)
        {
           printf("%d ",arr[i][j]);
        }
        printf("\n ");
    }
		FindMax();
		printf("�ռ���·��Ϊ��\n");
		HuiSu();
    return 0;
}

int Max(int a,int b)  //��ϴ�ֵ����
{
	return a>=b? a:b;
}

void FindMax()
{
	for(int i=1;i<=m;i++)
	{
		for(int j=1;j<=n;j++)
		{
			F[i][j]=Max(F[i-1][j],F[i][j-1])+arr[i][j];
		}
	}
		printf("\n����Ѽ� %d ��Ӳ�ҡ�\n",F[m][n]);
}


void HuiSu()//���ݣ����ҵ��ﵱǰλ�ã��Ǵ���߻����������ġ�
{
	int a=m;
	int b=n;
	H[1][1]=1;  //�����յ㶼Ϊ�ؾ����ĵ�
	H[m][n]=1;  //
	while(a>=1&&b>=1)
	{
		if(F[a-1][b]>=F[a][b-1])
		{
			H[a-1][b]=1;
			a--;  //�������һ��˵�����������
		}
		else
		{
			H[a][b-1]=1;
			b--;  //�������һ��˵�����Ϲ�����
		}
	}
	for(int i=1;i<=m;i++)
		for(int j=1;j<=n;j++)
		{
			if(H[i][j]==1)
			{
				if(i==m&&j==n)
					printf("��%d��%d��\n",m,n); 
				else
					printf("��%d��%d��-->",i,j);
			}
		}
}


