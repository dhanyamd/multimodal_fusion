from cv2.gapi import video
import torch 
import torch.nn as nn 
from transformers import BertModel, PretrainedBartModel 
from torchvision import models as vision_models
from sklearn.metrics import precision_score, accuracy_score
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

from meld_dataset import MELDDataset

class TextEncoder(nn.Module): 
    def __init__(self):
        super().__init__() 
        self.bert = BertModel.from_pretrained('bert-base-uncased') 

        for param in self.bert.parameters(): 
            param.requires_grad = False 
        self.projection = nn.Linear(768,128) 
    
    def forward(self, input_ids, attention_mask): 
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs.pooler_output 
        return self.projection(pooler_output) 

class VideoEncoder(nn.Module): 
    def __init__(self):
        super().__init__()
        self.backbone = vision_models.video.r3d_18(pretrained=True) 

        for params in self.backbone.parameters(): 
            params.requires_grad = False 
        num_fts = self.backbone.fc.in_features
        self.bacbone.fc = nn.Sequential(
            nn.Linear(num_fts, 128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
    def forward(self, x): 
        x = x.transpose(1,2) 
        return self.backbone(x)

class AudioEncoder(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.conv_layers = nn.Sequential(
            nn.Conv1d(64, 64, kernel_size=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64,128,kernel_size=3),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        for param in self.conv_layers.parameters(): 
            param.requires_grad=False 
        self.projection = nn.Sequential(
            nn.Linear(128,128),
            nn.Relu(),
            nn.Dropout(0.2)
        )
    def forward(self, x): 
        x = x.squeeze(1) 
        features = self.conv_layers(x) 
        return self.projection(features.squeeze(-1))

class MultimodalSentimentModel(nn.Module): 
    def __init__(self): 
        super().__init__() 
        self.text_encoder = TextEncoder()
        self.video_encoder = VideoEncoder() 
        self.audio_encoder = AudioEncoder() 

        #fusion layer 
        self.fusion_layer = nn.Sequential(
            nn.Linear(128 *3, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        #classification heads 
        self.emotion_classifier = nn.Sequential(
            nn.Linear(256,64),
            nn.Relu(),
            nn.Dropout(0.2),
            nn.Linear(64,7)
        )
        self.sentiment_classifier = nn.Sequential(
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 3)  # Negative, positive, neutral
        )
    def forward(self, text_inputs, video_frames, audio_frames): 
        text_features = self.text_encoder(
            text_inputs['input_ids'],
            text_inputs['attention_mask']
        )
        video_features = self.video_encoder(video_frames) 
        audio_features = self.audio_encoder(audio_frames) 

        combined_features = torch.concat([
            text_features,
            video_features,
            audio_features  
        ],dim=1) 
        fused_features = self.fusion_layer(combined_features) 

        emotion_output = self.emotion_classifier(fused_features)
        sentiment_output = self.sentiment_classifier(fused_features) 

        return {
            'emotions': emotion_output,
            'sentiment': sentiment_output
        }
    