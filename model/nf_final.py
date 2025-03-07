import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# 데이터 전처리 함수
def preprocess_data(df):
    """
    주어진 데이터프레임을 전처리하여, 범주형 변수에 대해 Label Encoding을 수행하고,
    수치형 변수에 대해 스케일링을 수행함.
    
    Args:
        df (DataFrame): 원본 데이터프레임

    Returns:
        df (DataFrame): 전처리된 데이터프레임
        encoders (dict): LabelEncoder 객체들을 포함하는 딕셔너리
    """
    # Label Encoding을 위한 Encoder 객체 생성
    encoders = {
        'user': LabelEncoder(),
        'isbn': LabelEncoder(),
        'author': LabelEncoder(),
        'title': LabelEncoder(),
        'publisher': LabelEncoder(),
        'country': LabelEncoder()
    }
    
    # 각 범주형 변수를 Label Encoding 수행
    df['user_idx'] = encoders['user'].fit_transform(df['id'])  # 사용자 id를 숫자로 변환
    df['item_idx'] = encoders['isbn'].fit_transform(df['isbn'])  # ISBN을 숫자로 변환
    df['author_idx'] = encoders['author'].fit_transform(df['author'])  # 저자 이름을 숫자로 변환
    df['title_idx'] = encoders['title'].fit_transform(df['title'])  # 책 제목을 숫자로 변환
    df['publisher_idx'] = encoders['publisher'].fit_transform(df['publisher'])  # 출판사명을 숫자로 변환
    df['country_idx'] = encoders['country'].fit_transform(df['country'])  # 국가를 숫자로 변환
    
    # 수치형 변수(나이, 출판년도)의 스케일링
    age_min, age_max = df['age'].min(), df['age'].max()  # 나이의 최소, 최대값 계산
    year_min, year_max = df['year'].min(), df['year'].max()  # 출판년도 최소, 최대값 계산
    
    # Min-Max 스케일링 (0~1로 변환)
    df['age_scaled'] = (df['age'] - age_min) / (age_max - age_min)  # 나이를 0~1 사이로 스케일링
    df['year_scaled'] = (df['year'] - year_min) / (year_max - year_min)  # 출판년도를 0~1 사이로 스케일링

    return df, encoders  # 전처리된 데이터프레임과 LabelEncoder 객체 반환


# Dataset 클래스 정의
class BookRatingDataset(Dataset):
    """
    책의 평점 데이터를 PyTorch의 Dataset 클래스로 감싸는 클래스입니다.
    각 컬럼을 tensor 형태로 반환하여 DataLoader에서 배치로 사용할 수 있습니다.
    """
    def __init__(self, df):
        self.df = df
        
        # 각 컬럼을 numpy 배열로 변환
        self.user_idx = self.df['user_idx'].values  # 사용자 인덱스 배열
        self.item_idx = self.df['item_idx'].values  # 아이템(책) 인덱스 배열
        self.age_scaled = self.df['age_scaled'].values  # 스케일링된 나이 배열
        self.year_scaled = self.df['year_scaled'].values  # 스케일링된 출판년도 배열
        self.country_idx = self.df['country_idx'].values  # 국가 인덱스 배열
        self.title_idx = self.df['title_idx'].values  # 책 제목 인덱스 배열
        self.author_idx = self.df['author_idx'].values  # 저자 인덱스 배열
        self.publisher_idx = self.df['publisher_idx'].values  # 출판사 인덱스 배열
        self.ratings = self.df['rating'].values  # 평점 배열

    def __len__(self):
        # 데이터의 샘플 수를 반환
        return len(self.df)
    
    def __getitem__(self, idx):
        # 주어진 인덱스(idx)로 데이터를 가져오는 함수
        return (
            torch.tensor(self.user_idx[idx], dtype=torch.long),  # 사용자 인덱스를 tensor로 변환
            torch.tensor(self.item_idx[idx], dtype=torch.long),  # 아이템 인덱스를 tensor로 변환
            torch.tensor(self.age_scaled[idx], dtype=torch.float),  # 나이 스케일링된 값을 tensor로 변환
            torch.tensor(self.year_scaled[idx], dtype=torch.float),  # 출판년도 스케일링된 값을 tensor로 변환
            torch.tensor(self.title_idx[idx], dtype=torch.long),  # 책 제목 인덱스를 tensor로 변환
            torch.tensor(self.country_idx[idx], dtype=torch.long),  # 국가 인덱스를 tensor로 변환
            torch.tensor(self.author_idx[idx], dtype=torch.long),  # 저자 인덱스를 tensor로 변환
            torch.tensor(self.publisher_idx[idx], dtype=torch.long),  # 출판사 인덱스를 tensor로 변환
            torch.tensor(self.ratings[idx], dtype=torch.float)  # 평점을 tensor로 변환
        )


# Extended NeuMF 모델 정의 (GMF + MLP)
class ExtendedNeuMF(nn.Module):
    def __init__(self, num_users, num_items, num_authors, num_publishers, num_titles,
                 num_countries, embedding_size=8, layers=[64,32,16],
                 dropout_rate=0.3, l2_reg=0.01):
        """
        Args:
            num_users (int): 사용자 수
            num_items (int): 아이템 수
            num_authors (int): 저자 수
            num_publishers (int): 출판사 수
            num_titles (int): 제목 수
            num_countries (int): 국가 수
            embedding_size (int): 임베딩 차원 크기
            layers (list): MLP 모델 레이어 크기
            dropout_rate (float): 드롭아웃 비율
            l2_reg (float): L2 정규화 계수
        """
        super(ExtendedNeuMF, self).__init__()
        self.embedding_size = embedding_size  # 임베딩 크기 설정
        self.layers = layers  # MLP 레이어 크기 설정
        self.l2_reg = l2_reg  # L2 정규화 계수 설정
        
        # 범주형 변수의 임베딩 차원 정의
        self.embedding_size_categorical = embedding_size // 2
        
        # GMF 임베딩
        self.mf_user_embedding = nn.Embedding(num_users, self.embedding_size_categorical)  # 사용자 임베딩
        self.mf_item_embedding = nn.Embedding(num_items, self.embedding_size_categorical)  # 아이템 임베딩
        
        # MLP 임베딩
        self.mlp_user_embedding = nn.Embedding(num_users, self.embedding_size_categorical)  # 사용자 임베딩
        self.mlp_item_embedding = nn.Embedding(num_items, self.embedding_size_categorical)  # 아이템 임베딩
        
        # 추가 범주형 변수(저자, 출판사, 국가)의 임베딩
        self.author_embedding = nn.Embedding(num_authors, self.embedding_size_categorical)  # 저자 임베딩
        self.publisher_embedding = nn.Embedding(num_publishers, self.embedding_size_categorical)  # 출판사 임베딩
        self.country_embedding = nn.Embedding(num_countries, self.embedding_size_categorical)  # 국가 임베딩
        self.title_embedding = nn.Embedding(num_titles, self.embedding_size_categorical)  # 제목 임베딩
        
        # MLP 입력 크기 계산: user, item 및 추가 범주형 변수를 포함
        mlp_input_size = self.embedding_size_categorical * 6  # 사용자, 아이템, 저자, 출판사, 국가, 제목
        mlp_input_size += 2  # 나이와 출판년도
        
        # MLP 레이어 정의
        input_size = mlp_input_size
        self.mlp_layers = nn.ModuleList()  # MLP 레이어 리스트
        self.skip_layers = nn.ModuleList()  # Skip connection 레이어 리스트
        for layer_size in layers:
            self.mlp_layers.append(nn.Sequential(
                nn.Linear(input_size, layer_size),  # 레이어 입력 크기
                nn.GELU(),  # 활성화 함수
                nn.Dropout(dropout_rate),  # 드롭아웃
                nn.LayerNorm(layer_size)  # 레이어 정규화
            ))
            
            # Skip connection
            if input_size != layer_size:
                self.skip_layers.append(nn.Sequential(
                    nn.Linear(input_size, layer_size),  # skip connection을 위한 선형 레이어
                    nn.Dropout(dropout_rate)  # 드롭아웃
                ))
            else:
                self.skip_layers.append(nn.Identity())  # 동일한 차원의 경우 그대로 연결
                
            input_size = layer_size  # 다음 레이어의 입력 크기
        
        # 최종 예측 레이어
        self.prediction = nn.Linear(self.embedding_size_categorical + layers[-1], 1)  # 예측을 위한 출력 레이어
        self._init_weights()  # 가중치 초기화

    def _init_weights(self):
        # Xavier 초기화로 가중치 초기화
        for m in self.modules():
            if isinstance(m, nn.Linear):  # Linear 레이어일 경우
                nn.init.xavier_uniform_(m.weight, gain=0.5)  # Xavier 초기화
                if m.bias is not None:
                    nn.init.zeros_(m.bias)  # Bias는 0으로 초기화
            elif isinstance(m, nn.Embedding):  # Embedding 레이어일 경우
                nn.init.normal_(m.weight, std=0.01)  # 정규 분포로 초기화
    
    def get_l2_reg_loss(self):
        # L2 정규화 손실 계산
        l2_loss = 0.0
        for param in self.parameters():  # 모든 파라미터에 대해
            l2_loss += torch.norm(param, 2)  # L2 노름 계산
        return self.l2_reg * l2_loss  # L2 정규화 손실 반환

    def forward(self, user_input, item_input, age_input, year_input, title_input,
                country_input, author_input, publisher_input):
        # GMF 파트: 사용자와 아이템의 내적
        mf_user_latent = self.mf_user_embedding(user_input)  # 사용자 임베딩
        mf_item_latent = self.mf_item_embedding(item_input)  # 아이템 임베딩
        mf_vector = torch.mul(mf_user_latent, mf_item_latent)  # 내적

        # MLP 파트: 임베딩된 모든 정보를 결합하여 예측
        mlp_user_latent = self.mlp_user_embedding(user_input)  # 사용자 임베딩
        mlp_item_latent = self.mlp_item_embedding(item_input)  # 아이템 임베딩
        author_latent = self.author_embedding(author_input)  # 저자 임베딩
        publisher_latent = self.publisher_embedding(publisher_input)  # 출판사 임베딩
        country_latent = self.country_embedding(country_input)  # 국가 임베딩
        title_latent = self.title_embedding(title_input)  # 제목 임베딩
        
        # 나이와 출판년도
        age_input = age_input.view(-1, 1)  # 나이의 차원을 2D로 변환
        year_input = year_input.view(-1, 1)  # 출판년도 차원을 2D로 변환
        
        # MLP 입력 결합
        mlp_vector = torch.cat([
            mlp_user_latent,
            mlp_item_latent,
            author_latent,
            publisher_latent,
            country_latent,
            title_latent,
            age_input,
            year_input
        ], dim=1)  # 여러 입력 벡터를 연결

        # MLP 레이어 및 skip connection 처리
        skip_input = mlp_vector
        for layer, skip in zip(self.mlp_layers, self.skip_layers):
            mlp_vector = layer(mlp_vector)  # 레이어 출력
            skip_output = skip(skip_input)  # skip connection 출력
            mlp_vector = mlp_vector + skip_output  # skip connection 더하기
            skip_input = mlp_vector  # skip_input 갱신
            
        # GMF와 MLP 벡터 결합 후 최종 예측
        predict_vector = torch.cat([mf_vector, mlp_vector], dim=1)  # GMF와 MLP 벡터 결합
        prediction = self.prediction(predict_vector)  # 최종 예측
        
        return prediction  # 예측 결과 반환




def create_data_loaders(df, train_ratio=0.7, batch_size=32):
    """전처리된 데이터를 train/test로 분할하고 DataLoader를 생성"""
    total_size = len(df)  # 데이터프레임의 총 크기 (샘플 수)
    indices = np.random.permutation(total_size)  # 전체 데이터를 무작위로 섞은 인덱스 배열 생성
    
    train_size = int(train_ratio * total_size)  # 훈련 데이터 크기 계산 (전체 크기의 70%)
    
    # 훈련 데이터와 테스트 데이터의 인덱스 분할
    train_indices = indices[:train_size]  # 훈련 데이터의 인덱스
    test_indices = indices[train_size:]  # 테스트 데이터의 인덱스
    
    # 훈련 데이터와 테스트 데이터로 데이터프레임 분리
    train_df = df.iloc[train_indices].reset_index(drop=True)  # 훈련 데이터
    print(len(train_df))  # 훈련 데이터의 크기 출력

    test_df = df.iloc[test_indices].reset_index(drop=True)  # 테스트 데이터
    print(len(test_df))  # 테스트 데이터의 크기 출력
    
    # 훈련 데이터와 테스트 데이터를 사용하여 Dataset 객체 생성
    train_dataset = BookRatingDataset(train_df)
    test_dataset = BookRatingDataset(test_df)
    
    # DataLoader 생성: 배치 단위로 데이터를 불러오는 역할
    train_loader = DataLoader(
        train_dataset,  # 훈련 데이터셋
        batch_size=batch_size,  # 배치 크기
        shuffle=True  # 데이터를 섞어서 불러오기
    )
    
    test_loader = DataLoader(
        test_dataset,  # 테스트 데이터셋
        batch_size=batch_size,  # 배치 크기
        shuffle=False  # 테스트 데이터는 섞지 않음
    )
    
    return train_loader, test_loader  # 훈련과 테스트용 DataLoader 반환

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4, verbose=True):
        """
        Early Stopping 클래스
        
        Args:
            patience (int): 성능 향상이 없어도 학습을 지속하는 에폭 수
            min_delta (float): 성능 향상으로 간주할 최소 변화량
            verbose (bool): 로그 출력 여부
        """
        self.patience = patience  # 성능 향상이 없을 때 중지 여부를 판단할 에폭 수
        self.min_delta = min_delta  # 성능 변화가 이 값보다 작으면 향상으로 간주하지 않음
        self.verbose = verbose  # 로그 출력 여부
        self.counter = 0  # 성능 향상이 없는 연속적인 에폭 수를 추적
        self.best_loss = None  # 가장 낮은 검증 손실 값
        self.early_stop = False  # 얼리 스토핑이 트리거됐는지 여부
        self.val_loss_min = float('inf')  # 초기 검증 손실 값 (무한대로 설정)
    
    def __call__(self, val_loss, model, epoch):
        """
        검증 손실(val_loss)을 비교하여 얼리 스토핑을 판단하는 메소드
        """
        if self.best_loss is None:  # 초기에는 가장 낮은 손실 값을 설정
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)  # 모델 저장
        elif val_loss > self.best_loss - self.min_delta:  # 성능이 향상되지 않으면 카운트를 증가시킴
            self.counter += 1
            if self.verbose:  # verbose가 True일 때만 로그 출력
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:  # patience 수를 초과하면 학습 중지
                self.early_stop = True
        else:  # 성능이 향상되면 가장 낮은 손실 값을 업데이트하고 카운트를 리셋
            self.best_loss = val_loss
            self.save_checkpoint(val_loss, model, epoch)  # 모델 저장
            self.counter = 0  # 카운트 초기화
    
    def save_checkpoint(self, val_loss, model, epoch):
        '''검증 손실이 감소할 때마다 모델을 저장'''
        if self.verbose:  # verbose가 True일 때만 로그 출력
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save({
            'epoch': epoch,  # 현재 에폭
            'model_state_dict': model.state_dict(),  # 모델의 상태 딕셔너리
            'loss': val_loss,  # 현재 검증 손실
        }, 'checkpoint.pt')  # 'checkpoint.pt'로 모델 저장
        self.val_loss_min = val_loss  # 최적의 검증 손실 값 갱신

def train_and_evaluate(train_loader, test_loader, model, criterion, optimizer, scheduler, device, epochs=20):
    best_train_rmse = float('inf')  # best 훈련 RMSE
    early_stopping = EarlyStopping(patience=10, verbose=True)  # Early Stopping 초기화
    
    print("Training Started...")
    for epoch in range(epochs):  # 주어진 에폭 수만큼 훈련을 반복
        # 학습
        model.train()  # 모델을 학습 모드로 설정
        total_loss = 0  # 총 손실값 초기화
        for batch in train_loader:  # 훈련 데이터로 배치 단위로 반복
            # 배치 데이터를 device에 맞게 전달
            user_input, item_input, age_input, year_input, title_input, country_input, author_input, publisher_input, ratings = [x.to(device) for x in batch]
            
            optimizer.zero_grad()  # 기울기 초기화
            
            # 모델 예측
            predictions = model(
                user_input, item_input, age_input, year_input, title_input,
                country_input, author_input, publisher_input
            )
            
            # 손실 계산
            loss = criterion(predictions, ratings.float().view(-1, 1))
            loss.backward()  # 역전파
            optimizer.step()  # 가중치 갱신
            
            total_loss += loss.item()  # 총 손실에 현재 배치 손실 추가
        
        current_rmse = total_loss / len(train_loader)  # 현재 훈련 RMSE 계산
        
        # 검증 손실 계산
        model.eval()  # 모델을 평가 모드로 설정
        val_loss = 0  # 검증 손실 초기화
        with torch.no_grad():  # 검증 단계에서는 기울기를 계산하지 않음
            for batch in test_loader:  # 테스트 데이터로 배치 단위로 반복
                user_input, item_input, age_input, year_input, title_input, country_input, author_input, publisher_input, ratings = [x.to(device) for x in batch]
                
                predictions = model(
                    user_input, item_input, age_input, year_input, title_input,
                    country_input, author_input, publisher_input
                )
                
                loss = criterion(predictions, ratings.float().view(-1, 1))  # 손실 계산
                val_loss += loss.item()  # 검증 손실 누적
        
        val_rmse = val_loss / len(test_loader)  # 검증 RMSE 계산
        
        # Learning rate 스케줄러 step
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(val_rmse)  # ReduceLROnPlateau의 경우 검증 RMSE로 학습률 조정
        else:
            scheduler.step()  # 그 외 스케줄러는 에폭 수에 따라 학습률 조정
            
        current_lr = optimizer.param_groups[0]['lr']  # 현재 학습률 출력
        print(f'Epoch {epoch+1}/{epochs} - Train RMSE: {current_rmse:.4f}, Val RMSE: {val_rmse:.4f}, Learning Rate: {current_lr:.6f}')
        
        if current_rmse < best_train_rmse:  # 훈련 RMSE가 최적이면 갱신
            best_train_rmse = current_rmse
            
        # Early Stopping 체크
        early_stopping(val_rmse, model, epoch)  # 검증 손실을 바탕으로 얼리 스토핑 체크
        if early_stopping.early_stop:  # 얼리 스토핑이 트리거되면 학습 중단
            print("Early stopping triggered")
            break
    
    print("\nTraining Completed!")
    print(f"Best Train RMSE: {best_train_rmse:.4f}")
    
    # 최적의 모델 로드
    checkpoint = torch.load('checkpoint.pt')  # 체크포인트 로드
    model.load_state_dict(checkpoint['model_state_dict'])  # 모델 상태 딕셔너리 로드
    
    # 최종 테스트 평가
    print("\nEvaluating on Test Set...")
    model.eval()  # 모델을 평가 모드로 설정ㄴ
    test_loss = 0  # 테스트 손실 초기화
    with torch.no_grad():  # 기울기 계산 안 함
        for batch in test_loader:  # 테스트 데이터로 배치 단위로 반복
            user_input, item_input, age_input, year_input, title_input, country_input, author_input, publisher_input, ratings = [x.to(device) for x in batch]
            
            predictions = model(
                user_input, item_input, age_input, year_input, title_input,
                country_input, author_input, publisher_input
            )
            
            loss = criterion(predictions, ratings.float().view(-1, 1))  # 손실 계산
            test_loss += loss.item()  # 테스트 손실 누적
    
    final_test_rmse = test_loss / len(test_loader)  # 최종 테스트 RMSE 계산
    print(f"Final Test RMSE: {final_test_rmse:.4f}")
    
    return best_train_rmse, final_test_rmse  # 훈련 RMSE와 테스트 RMSE 반환


'''RMSELossWithSmoothing
: 실제 평점(target)을 약간 부드럽게 만드는 기법.
epsilon 비율만큼 원래 평점을 평균 평점 쪽으로 이동시킨다.
예: 실제 평점이 10점일 때 epsilon=0.1이면

90%는 원래 평점(10점), 10%는 평균 평점((1+10)/2 = 5.5점) => 최종 평점 = 0.9 * 10 + 0.1 * 5.5 = 9.45점
모델이 너무 극단적인 값을 예측하지 않도록 도와줌.'''
class RMSELossWithSmoothing(nn.Module):
    def __init__(self, epsilon=0.1, min_rating=1, max_rating=10):
        super().__init__()
        self.epsilon = epsilon  # smoothing 강도
        self.min_rating = min_rating  # 최소 평점
        self.max_rating = max_rating  # 최대 평점
        
    def forward(self, pred, target):
        # target을 smoothing
        # 원래 타겟과 균일 분포 사이의 가중 평균을 계산
        smooth_target = (1 - self.epsilon) * target + self.epsilon * (self.min_rating + self.max_rating) / 2
        
        # RMSE 계산
        mse = torch.mean((pred - smooth_target) ** 2)
        rmse = torch.sqrt(mse)
        
        return rmse

 


if __name__ == "__main__":
    # 데이터 로드
    df = pd.read_csv("전처리된 csv파일 경로 입력", index_col=0)
    
    # 전처리 수행
    df, encoders = preprocess_data(df)
    
    # 하이퍼파라미터 설정
    EMBEDDING_SIZE = 32
    LAYERS = [32, 16, 8, 4]
    LEARNING_RATE = 0.005
    BATCH_SIZE = 32
    EPOCHS = 20
    DROPOUT_RATE = 0.2
    
    # 디바이스 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    
    # 데이터로더 생성
    train_loader, test_loader = create_data_loaders(
        df,
        train_ratio=0.7,
        batch_size=BATCH_SIZE
    )
    
    # 모델 차원 정보 얻기
    num_users = len(encoders['user'].classes_)
    num_items = len(encoders['isbn'].classes_)
    num_authors = len(encoders['author'].classes_)
    num_publishers = len(encoders['publisher'].classes_)
    num_countries = len(encoders['country'].classes_)
    num_titles = len(encoders['title'].classes_)
    
    # 모델 초기화

    model = ExtendedNeuMF(
        num_users=num_users,
        num_items=num_items,
        num_authors=num_authors,
        num_publishers=num_publishers,
        num_countries=num_countries,
        num_titles=num_titles,
        embedding_size=EMBEDDING_SIZE,
        layers=LAYERS,
        dropout_rate = DROPOUT_RATE
    ).to(device)
    
    # 손실 함수와 옵티마이저 설정
    criterion = RMSELossWithSmoothing(epsilon=0.2)  # epsilon은 smoothing 강도를 조절하는 하이퍼파라미터
    #criterion = RMSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=5,    # 5 에폭마다
        gamma=0.7       # 학습률을 0.5배로 감소
    )
    
    # 학습 및 평가
    best_train_rmse, final_test_rmse = train_and_evaluate(
        train_loader,
        test_loader,
        model,
        criterion,
        optimizer,
        scheduler,  # scheduler 추가
        device,
        EPOCHS
    )
