#######################
# Import libraries
import streamlit as st
import plotly.express as px
import pandas as pd
import geopandas as gpd
import folium
from folium.plugins import HeatMap
import json
from IPython.display import display
import streamlit.components.v1 as components
from folium import plugins
import matplotlib.pyplot as plt
from shapely.geometry import Polygon
import h3
import libpysal as lps
from libpysal.weights import Queen 
import esda
from splot.esda import plot_moran, moran_scatterplot, lisa_cluster
from esda.moran import Moran, Moran_Local
from streamlit_folium import folium_static
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.colors as mcolors
from folium import IFrame
from folium.plugins import Fullscreen, FloatImage
from folium.plugins import GroupedLayerControl
from PIL import Image
import io
from branca.element import Template, MacroElement
import branca

#######################
# Page configuration
st.set_page_config(
    page_title="국립공원 Dashboard",
    layout="wide",
    initial_sidebar_state="expanded")
#######################


# Plots
def find_boundary(npark_boundary,npark_name):
    npark_boundary = npark_boundary[npark_boundary['DESIG']=='국립공원']
    seoul_npark_boundary = npark_boundary[npark_boundary['ORIG_NAME']==npark_name]
    return seoul_npark_boundary

# Plots
def find_boundary_hotspot(npark_boundary,npark_name):
    npark_boundary = npark_boundary[npark_boundary['DESIG']=='국립공원']
    seoul_npark_boundary = npark_boundary[npark_boundary['ORIG_NAME']==npark_name]
    name_list = ['설악산','변산반도','경주','덕유산','다도해해상','월악산','오대산','한려해상','태안해안']
    if npark_name in name_list:
        seoul_npark_boundary=seoul_npark_boundary.explode()
        seoul_npark_boundary = seoul_npark_boundary.reset_index()
        seoul_npark_boundary = seoul_npark_boundary[seoul_npark_boundary.index==0]
    
    return seoul_npark_boundary

def sjoin(gdf_park_data,npark_boundary,npark_name):
    gdf_seoul_park_data = gdf_park_data[gdf_park_data['국립공원명']==npark_name]
    seoul_accident = gpd.sjoin(gdf_seoul_park_data, npark_boundary)
    return seoul_accident

def find_center_latitudes_longtitudes(accident):
    latitudes = accident['위도_변환'].tolist()
    longitudes = accident['경도_변환'].tolist()
    map_center_lat = sum(latitudes) / len(latitudes)
    map_center_lon = sum(longitudes) / len(longitudes)
    return map_center_lat,map_center_lon

def v_world(selected_national_park_accident):
    map_center_lat,map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)

    tiles = f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/{layer}/{{z}}/{{y}}/{{x}}.{tileType}"
    attr = "Vworld"
    # 기본 지도 객체 생성
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12,tiles=tiles, attr=attr)
    # VWorld Hybrid 타일 추가
    satelitelayer = folium.TileLayer(
        tiles=f'http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png',
        attr='VWorld Hybrid',
        name='지명표시',
        overlay=True
    ).add_to(m)
    return m


def make_pointplot(selected_national_park_accident,selected_npark_boundary):    

    
    map_center_lat, map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Hybrid 타일 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)
    # 사고 원인별 색상 사전 정의
    palette = sns.color_palette('bright')

    # 사건 원인에 대한 색상 딕셔너리
    color_dict = {
        '실족ㆍ골절': palette[0],
        '기타': palette[1],
        '일시적고립': palette[2],
        '탈진경련': palette[3],
        '낙석ㆍ낙빙': palette[4],
        '추락': palette[5],
        '심장사고': palette[6],
        '해충피해': palette[7],
        '익수': palette[8],
    }

    # 컬러 팔레트에 해당하는 RGB 값을 hex 코드로 변환
    color_dict_hex = {key: mcolors.rgb2hex(value) for key, value in color_dict.items()}
# 사고 원인별로 레이어 그룹 생성 및 추가
    groups = {'사고 원인': []}  # 사고 원인별 그룹을 담을 리스트를 생성합니다.
        # 모든 사고 원인을 담을 FeatureGroup을 생성합니다.
    all_accidents_group = folium.FeatureGroup(name="전체 사고 원인")

    for i, color in color_dict_hex.items():
        type_accident = selected_national_park_accident[selected_national_park_accident['유형'] == i]
        accident_color = color  # 사고 원인별로 정의된 색상 사용
        feature_group = folium.FeatureGroup(name=i)  # 사고 원인별 FeatureGroup 생성

        # 사고 위치에 대한 CircleMarker 추가 및 팝업 정보 설정
        for idx, row in type_accident.iterrows():
            tooltip_text =f"유형: {row['유형']}<br>사고 일자: {row['연월일']}<br>위치: {row['위도_변환']}, {row['경도_변환']}<br>사고장소: {row['위치']}<br>고도: {float(row['고도']):.2f}<br>경사도: {float(row['경사도']):.2f}"
            folium.CircleMarker(
                location=(row['위도_변환'], row['경도_변환']),
                radius=3,
                color=accident_color,
                fill=True,
                fill_color=accident_color,
                fill_opacity=1,  # 내부 채움 불투명도
                tooltip=tooltip_text,
                
            ).add_to(feature_group)
            
        feature_group.add_to(all_accidents_group)  # FeatureGroup을 지도 객체에 추가
        groups['사고 원인'].append(feature_group)  # 사고 원인별로 그룹에 FeatureGroup을 추가합니다.

    # 전체 사고 원인 그룹을 지도에 추가합니다.
    all_accidents_group.add_to(m)
    # 사고 원인별 그룹을 그룹화된 레이어 컨트롤로 추가
    GroupedLayerControl(groups=groups, collapsed=False, exclusive_groups=False).add_to(m)

    folium.LayerControl().add_to(m)


    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:13px; left: 10px; bottom: 10px;'>
        
    <div class='legend-title'>사고원인 범례</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
        <li><span style='background:blue;opacity:1;'></span><strong>실족ㆍ골절</strong></li>
        <li><span style='background:#FC9238;opacity:1;'></span><strong>기타</strong></li>
        <li><span style='background:#75D456;opacity:1;'></span><strong>일시적고립</strong></li>
        <li><span style='background:#C82E23;opacity:1;'></span><strong>탈진경련</strong></li>
        <li><span style='background:#8E44AD;opacity:1;'></span><strong>낙석ㆍ낙빙</strong></li>
        <li><span style='background:#784315;opacity:1;'></span><strong>추락</strong></li>
        <li><span style='background:#E05EC6;opacity:1;'></span><strong>심장사고</strong></li>
        <li><span style='background:gray;opacity:1;'></span><strong>해충피해</strong></li>
        <li><span style='background:#F4D03F;opacity:1;'></span><strong>익수</strong></li>




    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Add the legend to the map
    macro = MacroElement()
    macro._template = Template(template)
    macro.add_to(m)



    return m,color_dict_hex



########################################### 핫스팟과 안전쉼터가 입력한 거리보다 넘어간 핫스팟 위치 표시 ##################################
def make_hotspot_safetyplace(selected_national_park_accident,selected_npark_boundary,safety_place,distance=500):
    single_polygon = selected_npark_boundary['geometry'].unary_union
    # GeoJSON 형식으로 변환
    geojson_polygon = single_polygon.__geo_interface__
    resolution=9
    # 북한산 국립공원을 커버하는 H3 육각형 인덱스 생성
    hexagons = list(h3.polyfill_geojson(geojson_polygon, resolution))

    # polygon = seoul_npark_boundary.iloc[0]['geometry']
    # 육각형 경계 좌표 생성
    hexagon_polygons = [Polygon(h3.h3_to_geo_boundary(hex, geo_json=True)) for hex in hexagons]

    # GeoDataFrame 생성
    gdf_polygon = gpd.GeoDataFrame(geometry=hexagon_polygons)

    prices = gpd.sjoin(gdf_polygon, selected_national_park_accident[['geometry']], op='contains')
    
    # 각 geometry에 대한 카운트를 계산
    nbr_avg_price = prices.groupby('geometry').size().reset_index(name='acc_counts')
    # 병합하기
    nbr_final = gdf_polygon.merge(nbr_avg_price, on='geometry', how='left')
    # 병합 결과에서 결측값을 0으로 채우기
    nbr_final['acc_counts'] = nbr_final['acc_counts'].fillna(0)
    w =  lps.weights.Queen.from_dataframe(nbr_final)
    w.transform = 'r'

    # 속성 유사성(Attribute similarity) / 가중치 적용 사고발생건수 추가
    nbr_final['weighted_acc_counts'] = lps.weights.lag_spatial(w, nbr_final['acc_counts'])

    # 광역적 공간 자기상관 / 상관계수값 출력 
    y = nbr_final.acc_counts
    moran = esda.Moran(y, w)
    moran_local = Moran_Local(y, w)
    nbr_final['Cluster'] = moran_local.q
    cluster_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    nbr_final['Cluster_Label'] = nbr_final['Cluster'].map(cluster_map)
    moran_local = Moran_Local(y, w)
    # p-value 기준으로 유의미한 결과만 필터링
    p_threshold = 0.05  # 유의수준 설정
    nbr_final['Significant'] = moran_local.p_sim < p_threshold

    # 유의미하지 않은 관측치에 대해 'NS' 라벨 할당
    nbr_final['Cluster_Label'] = nbr_final.apply(lambda row: 'NS' if not row['Significant'] else row['Cluster_Label'], axis=1)

    # 'Significant' 열은 이제 필요 없으므로 제거하거나, 유지하려면 이 단계를 생략
    nbr_final.drop(columns=['Significant'], inplace=True)

    center = nbr_final.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)

    # 클러스터 레이어 설정
    cluster_colors_핫스팟 = {
        'HH': 'red',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'transparent',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function_핫스팟(feature):
        return {
            'fillColor': cluster_colors_핫스팟.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"

    # 클러스터 레이어 설정
    cluster_colors_콜드스팟 = {
        'HH': 'transparent',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'blue',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function_콜드스팟(feature):
        return {
            'fillColor': cluster_colors_콜드스팟.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"



    # 클러스터 레이어 추가
    cluster_layer_핫스팟 = folium.FeatureGroup(name='전체사고 핫스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function_핫스팟,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['사고수:']
        )
    ).add_to(cluster_layer_핫스팟)
    cluster_layer_핫스팟.add_to(m)


    # 클러스터 레이어 추가
    cluster_layer_콜드스팟 = folium.FeatureGroup(name='전체사고 콜드스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function_콜드스팟,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['사고수:']
        )
    ).add_to(cluster_layer_콜드스팟)
    cluster_layer_콜드스팟.add_to(m)
    
    # # 안전쉼터 레이어 설정 및 추가
    # shelter_layer = folium.FeatureGroup(name='기존 안전쉼터 설치지점')
    # for idx, row in safety_place.iterrows():
    #     folium.CircleMarker(
    #         location=(row['위도'], row['경도']),
    #         popup=row['쉼터명'],
    #         radius=3,
    #         color='green',
    #         fill=True,
    #         fill_color='green',
    #         fill_opacity=1
    #     ).add_to(shelter_layer)
    # shelter_layer.add_to(m)

    # 안전쉼터 위치를 Marker로 추가
    shelter_layer = folium.FeatureGroup(name='안전쉼터')
    for idx, row in shelter.iterrows():
        # 팝업을 위한 HTML 컨텐츠 생성
        popup_html = f"""
        <div style='font-family: "Arial", sans-serif; font-size: 12px;'>
            <h4>{row['쉼터명']}</h4>
            <p>위도: {row['geometry'].y:.5f}<br>
            경도: {row['geometry'].x:.5f}</p>
        </div>
        """
        popup = folium.Popup(popup_html, max_width=250)

        # "대피소"가 쉼터명에 포함되어 있으면 마커 색을 'black'으로 설정
        icon_color = 'orange' if '대피소' in row['쉼터명'] else 'blue'

        # 마커 생성 (Font Awesome 아이콘 사용)
        Marker(
            location=[row['geometry'].y, row['geometry'].x],  # GeoDataFrame에서 위도, 경도 추출
            icon=Icon(icon='person-shelter', prefix='fa', color=icon_color),  # 조건에 따라 색상 변경
            popup=popup
        ).add_to(shelter_layer)

    shelter_layer.add_to(m)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # # 위치표지판 레이어 설정 및 추가
    # sign_layer = folium.FeatureGroup(name='다목적위치표지판')
    # for idx, row in sign_place.iterrows():
    #     folium.CircleMarker(
    #         location=(row['위도'], row['경도']),
    #         popup=row['위치'],
    #         radius=3,
    #         color='orange',
    #         fill=True,
    #         fill_color='orange',
    #         fill_opacity=0.8
    #     ).add_to(sign_layer)
    # sign_layer.add_to(m)





    def filter_hotspots_far_from_safetyplace(nbr_final, safety_place, min_distance=100, cluster_label='HH'):
        # 안전쉼터 데이터를 GeoDataFrame으로 변환
        safety_place_gdf = gpd.GeoDataFrame(
            safety_place,
            geometry=gpd.points_from_xy(safety_place.경도, safety_place.위도),
            crs='EPSG:4326'
        )

        # 좌표계 변환
        nbr_final_utm = nbr_final.to_crs(epsg=5174)
        safety_place_utm = safety_place_gdf.to_crs(epsg=5174)

        # 'HH' 클러스터 라벨이 지정된 핫스팟 선택
        nbr_final_핫스팟 = nbr_final_utm[nbr_final_utm['Cluster_Label'] == cluster_label]

                # nbr_final_핫스팟이 비어있는 경우 예외 발생
        if len(nbr_final_핫스팟) == 0:
            raise ValueError("No hotspots found with the specified cluster label.")


        # 각 사고 핫스팟에서 가장 가까운 안전쉼터까지의 최소 거리 계산
        def calculate_min_distance_to_AED(hotspot, shelters):
            # 사고 핫스팟과 모든 안전쉼터 사이의 거리 계산 후 최소값 반환
            distances = shelters.geometry.distance(hotspot.geometry)
            return distances.min()
        # 최소 거리 계산
        nbr_final_핫스팟['min_distance_to_shelter'] = nbr_final_핫스팟.apply(
            lambda hotspot: calculate_min_distance_to_AED(hotspot, safety_place_utm), axis=1
        )

        # 지정된 최소 거리 이상 떨어진 핫스팟 필터링
        hotspots_far_from_shelters = nbr_final_핫스팟[nbr_final_핫스팟['min_distance_to_shelter'] > min_distance]

        # WGS84 좌표계로 변환
        hotspots_far_from_shelters_wgs84 = hotspots_far_from_shelters.to_crs(epsg=4326)
        
        return hotspots_far_from_shelters_wgs84

    ########## 이격거리 조절가능 ########################## 
    out_hotspot_layer = folium.FeatureGroup(name='안전쉼터 추가설치 예측지점')
    try:
        for idx, row in filter_hotspots_far_from_safetyplace(nbr_final, safety_place, distance, 'HH').iterrows():
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                icon=folium.Icon(icon="home",color='green'),
            ).add_to(out_hotspot_layer)
        out_hotspot_layer.add_to(m)
    except ValueError as e:
        st.error("사고 발생 건수가 적어 지도 분석이 어려워요. 다른 공원을 분석해주세요.")

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)  

       # 사고 원인별 색상 사전 정의
    palette = sns.color_palette('bright')

    # 사건 유형에 대한 색상 딕셔너리
    color_dict = {
        '실족ㆍ골절': palette[0],
        '기타': palette[1],
        '일시적고립': palette[2],
        '탈진경련': palette[3],
        '낙석ㆍ낙빙': palette[4],
        '추락': palette[5],
        '심장사고': palette[6],
        '해충피해': palette[7],
        '익수': palette[8],
    }

    # 컬러 팔레트에 해당하는 RGB 값을 hex 코드로 변환
    color_dict_hex = {key: mcolors.rgb2hex(value) for key, value in color_dict.items()}
  
      # 사고 원인별로 레이어 그룹 생성 및 추가
    groups = {'사고 원인': []}  # 사고 원인별 그룹을 담을 리스트를 생성합니다.
        # 모든 사고 원인을 담을 FeatureGroup을 생성합니다.
    all_accidents_group = folium.FeatureGroup(name="전체 사고 원인")

    for i, color in color_dict_hex.items():
        type_accident = selected_national_park_accident[selected_national_park_accident['유형'] == i]
        accident_color = color  # 사고 원인별로 정의된 색상 사용
        feature_group = folium.FeatureGroup(name=i)  # 사고 원인별 FeatureGroup 생성

        # 사고 위치에 대한 CircleMarker 추가 및 팝업 정보 설정
        for idx, row in type_accident.iterrows():
            tooltip_text =f"유형: {row['유형']}<br>사고 일자: {row['연월일']}<br>위치: {row['위도_변환']}, {row['경도_변환']}<br>사고장소: {row['위치']}<br>고도: {float(row['고도']):.2f}<br>경사도: {float(row['경사도']):.2f}"
            folium.CircleMarker(
                location=(row['위도_변환'], row['경도_변환']),
                radius=3,
                color=accident_color,
                fill=True,
                fill_color=accident_color,
                fill_opacity=1,  # 내부 채움 불투명도
                tooltip=tooltip_text,
                
            ).add_to(feature_group)
            
        feature_group.add_to(all_accidents_group)  # FeatureGroup을 지도 객체에 추가
        groups['사고 원인'].append(feature_group)  # 사고 원인별로 그룹에 FeatureGroup을 추가합니다.


    # 전체 사고 원인 그룹을 지도에 추가합니다.
    all_accidents_group.add_to(m)
    # 사고 원인별 그룹을 그룹화된 레이어 컨트롤로 추가
    GroupedLayerControl(groups=groups, collapsed=False, exclusive_groups=False).add_to(m)
    

    # 레이어 컨트롤 추가하여 사용자가 레이어 선택 가능하게 함
    folium.LayerControl().add_to(m)


    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:13px; left: 10px; bottom: 150px;'>
        
    <div class='legend-title'>사고원인 범례</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
        <li><span style='background:blue;opacity:1;'></span><strong>실족ㆍ골절</strong></li>
        <li><span style='background:#E28332;opacity:1;'></span><strong>기타</strong></li>
        <li><span style='background:#75D456;opacity:1;'></span><strong>일시적고립</strong></li>
        <li><span style='background:#C82E23;opacity:1;'></span><strong>탈진경련</strong></li>
        <li><span style='background:#8E44AD;opacity:1;'></span><strong>낙석ㆍ낙빙</strong></li>
        <li><span style='background:#784315;opacity:1;'></span><strong>추락</strong></li>
        <li><span style='background:#E05EC6;opacity:1;'></span><strong>심장사고</strong></li>
        <li><span style='background:gray;opacity:1;'></span><strong>해충피해</strong></li>
        <li><span style='background:#F4D03F;opacity:1;'></span><strong>익수</strong></li>




    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Add the legend to the map
    macro = MacroElement()
    macro._template = Template(template)
    macro.add_to(m)

        
    template2 = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend2" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" integrity="sha384-ekzFgdUyJibDJSR+Jb+wdX4Axq7rKjHcXdMI1a3iFxoZou1GEGq3w5LaD1aMhohc" crossorigin="anonymous">
    </head>
    
    <div id='maplegend2' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:13px; left: 10px; bottom: 10px;'>
        
    <div class='legend-title'>범례</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
        <li><span style='background:#6EA8D8;opacity:1;'></span><strong>기존 안전쉼터 설치지점</strong></li>
        <li><span style='background:#83AE41;opacity:1;'></span><strong>안전쉼터 추가설치 예측지점</strong></li>
        <li><span style='background:#DC9948;opacity:1;'></span><strong>기존 대피소 설치지점</strong></li>

    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Add the legend to the map
    macro2 = MacroElement()
    macro2._template = Template(template2)
    macro2.add_to(m)

    return m















########################################### 핫스팟과 AED가 입력한 거리보다 넘어간 핫스팟 위치 표시 ##################################
def make_hotspot_heart(selected_national_park_accident,selected_npark_boundary,df_AED,distance=500):
    single_polygon = selected_npark_boundary['geometry'].unary_union
    # GeoJSON 형식으로 변환
    geojson_polygon = single_polygon.__geo_interface__
    resolution=9
    # 북한산 국립공원을 커버하는 H3 육각형 인덱스 생성
    hexagons = list(h3.polyfill_geojson(geojson_polygon, resolution))

    # polygon = seoul_npark_boundary.iloc[0]['geometry']
    # 육각형 경계 좌표 생성
    hexagon_polygons = [Polygon(h3.h3_to_geo_boundary(hex, geo_json=True)) for hex in hexagons]

    # GeoDataFrame 생성
    gdf_polygon = gpd.GeoDataFrame(geometry=hexagon_polygons)

    # 심장문제만 필터링
    selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['유형']=='심장사고']
    prices = gpd.sjoin(gdf_polygon, selected_national_park_accident[['geometry']], op='contains')

    # 각 geometry에 대한 카운트를 계산
    nbr_avg_price = prices.groupby('geometry').size().reset_index(name='acc_counts')
    # 병합하기
    nbr_final = gdf_polygon.merge(nbr_avg_price, on='geometry', how='left')
    # 병합 결과에서 결측값을 0으로 채우기
    nbr_final['acc_counts'] = nbr_final['acc_counts'].fillna(0)
    w =  lps.weights.Queen.from_dataframe(nbr_final)
    w.transform = 'r'

    # 속성 유사성(Attribute similarity) / 가중치 적용 사고발생건수 추가
    nbr_final['weighted_acc_counts'] = lps.weights.lag_spatial(w, nbr_final['acc_counts'])

    # 광역적 공간 자기상관 / 상관계수값 출력 
    y = nbr_final.acc_counts
    moran = esda.Moran(y, w)
    moran_local = Moran_Local(y, w)
    nbr_final['Cluster'] = moran_local.q
    cluster_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    nbr_final['Cluster_Label'] = nbr_final['Cluster'].map(cluster_map)
    moran_local = Moran_Local(y, w)
    # p-value 기준으로 유의미한 결과만 필터링
    p_threshold = 0.05  # 유의수준 설정
    nbr_final['Significant'] = moran_local.p_sim < p_threshold

    # 유의미하지 않은 관측치에 대해 'NS' 라벨 할당
    nbr_final['Cluster_Label'] = nbr_final.apply(lambda row: 'NS' if not row['Significant'] else row['Cluster_Label'], axis=1)

    # 'Significant' 열은 이제 필요 없으므로 제거하거나, 유지하려면 이 단계를 생략
    nbr_final.drop(columns=['Significant'], inplace=True)

    center = nbr_final.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)

    # 클러스터 레이어 설정
    cluster_colors_핫스팟 = {
        'HH': 'red',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'transparent',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function_핫스팟(feature):
        return {
            'fillColor': cluster_colors_핫스팟.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"

    # 클러스터 레이어 설정
    cluster_colors_콜드스팟 = {
        'HH': 'transparent',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'blue',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function_콜드스팟(feature):
        return {
            'fillColor': cluster_colors_콜드스팟.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"



    # 클러스터 레이어 추가
    cluster_layer_핫스팟 = folium.FeatureGroup(name='심장사고 핫스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function_핫스팟,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['심장사고수:']
        )
    ).add_to(cluster_layer_핫스팟)
    cluster_layer_핫스팟.add_to(m)


    # 클러스터 레이어 추가
    cluster_layer_콜드스팟 = folium.FeatureGroup(name='심장사고 콜드스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function_콜드스팟,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['심장사고수:']
        )
    ).add_to(cluster_layer_콜드스팟)
    cluster_layer_콜드스팟.add_to(m)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # # 위치표지판 레이어 설정 및 추가
    # sign_layer = folium.FeatureGroup(name='다목적위치표지판')
    # for idx, row in sign_place.iterrows():
    #     folium.CircleMarker(
    #         location=(row['위도'], row['경도']),
    #         popup=row['위치'],
    #         radius=3,
    #         color='orange',
    #         fill=True,
    #         fill_color='orange',
    #         fill_opacity=0.8
    #     ).add_to(sign_layer)
    # sign_layer.add_to(m)

    # AED 위치
    df_AED_layer = folium.FeatureGroup(name='기존 AED 설치지점')
    for idx, row in df_AED.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['명칭'],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1
        ).add_to(df_AED_layer)
    df_AED_layer.add_to(m)

    # 심장문제 사고지점 추가
    seoul_accident_heart_layer = folium.FeatureGroup(name='심장사고지점')
    for idx, row in selected_national_park_accident.iterrows():
            tooltip_text = f"유형: {row['유형']}<br>사고 일자: {row['연월일']}<br>위치: {row['위도_변환']}, {row['경도_변환']}<br>사고장소: {row['위치']}<br>고도: {float(row['고도']):.2f}<br>경사도: {float(row['경사도']):.2f}"                
            folium.CircleMarker(
            location=(row['위도_변환'], row['경도_변환']),
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1,
            tooltip=tooltip_text
        ).add_to(seoul_accident_heart_layer)
    seoul_accident_heart_layer.add_to(m)


    def filter_hotspots_far_from_AED(nbr_final, df_AED, min_distance=100, cluster_label='HH'):
        # 안전쉼터 데이터를 GeoDataFrame으로 변환
        df_AED_gdf = gpd.GeoDataFrame(
            df_AED,
            geometry=gpd.points_from_xy(df_AED.경도, df_AED.위도),
            crs='EPSG:4326'
        )

        # 좌표계 변환
        nbr_final_utm = nbr_final.to_crs(epsg=5174)
        df_AED_utm = df_AED_gdf.to_crs(epsg=5174)

        # 'HH' 클러스터 라벨이 지정된 핫스팟 선택
        nbr_final_핫스팟 = nbr_final_utm[nbr_final_utm['Cluster_Label'] == cluster_label]
        
        # nbr_final_핫스팟이 비어있는 경우 예외 발생
        if len(nbr_final_핫스팟) == 0:
            raise ValueError("No hotspots found with the specified cluster label.")

        # 각 사고 핫스팟에서 가장 가까운 안전쉼터까지의 최소 거리 계산
        def calculate_min_distance_to_shelter(hotspot, shelters):
            # 사고 핫스팟과 모든 안전쉼터 사이의 거리 계산 후 최소값 반환
            distances = shelters.geometry.distance(hotspot.geometry)
            return distances.min()

        # 최소 거리 계산
        nbr_final_핫스팟['min_distance_to_shelter'] = nbr_final_핫스팟.apply(
            lambda hotspot: calculate_min_distance_to_shelter(hotspot, df_AED_utm), axis=1
        )

        # 지정된 최소 거리 이상 떨어진 핫스팟 필터링
        hotspots_far_from_shelters = nbr_final_핫스팟[nbr_final_핫스팟['min_distance_to_shelter'] > min_distance]

        # WGS84 좌표계로 변환
        hotspots_far_from_shelters_wgs84 = hotspots_far_from_shelters.to_crs(epsg=4326)
        
        return hotspots_far_from_shelters_wgs84

    ########## 이격거리 조절가능 ########################## 
    out_hotspot_layer = folium.FeatureGroup(name='AED 추가설치 예측지점')
    try:
        for idx, row in filter_hotspots_far_from_AED(nbr_final, df_AED, distance, 'HH').iterrows():
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                icon=folium.Icon(icon="heart",color='pink'),
            ).add_to(out_hotspot_layer)
        out_hotspot_layer.add_to(m)
    except ValueError as e:
        st.error("심장사고 발생 건수가 적어 지도 분석이 어려워요. 다른 공원을 분석해주세요.")
#        # 사고 원인별 색상 사전 정의
#     palette = sns.color_palette('bright')

#     # 사건 유형에 대한 색상 딕셔너리
#     color_dict = {
#         '실족ㆍ골절': palette[0],
#         '기타': palette[1],
#         '일시적고립': palette[2],
#         '탈진경련': palette[3],
#         '낙석ㆍ낙빙': palette[4],
#         '추락': palette[5],
#         '심장사고': palette[6],
#         '해충피해': palette[7],
#         '익수': palette[8],
#     }

#     # 컬러 팔레트에 해당하는 RGB 값을 hex 코드로 변환
#     color_dict_hex = {key: mcolors.rgb2hex(value) for key, value in color_dict.items()}
#     # 사고 원인별로 레이어 그룹 생성 및 추가
#     # 사고 원인별로 레이어 그룹 생성 및 추가
#     accident_types = selected_national_park_accident['유형'].unique()
#    # 사고 원인별로 레이어 그룹 생성 및 추가
#     groups = {'사고 원인': []}  # 사고 원인별 그룹을 담을 리스트를 생성합니다.

#     for i, color in color_dict_hex.items():
#         type_accident = selected_national_park_accident[selected_national_park_accident['유형'] == i]
#         accident_color = color  # 사고 원인별로 정의된 색상 사용
#         feature_group = folium.FeatureGroup(name=i)  # 사고 원인별 FeatureGroup 생성

#         # 사고 위치에 대한 CircleMarker 추가 및 툴팁 정보 설정
#         for idx, row in type_accident.iterrows():
#             tooltip_text = f"유형: {row['유형']}<br>사고 일자: {row['연월일']}"  # 툴팁 텍스트 정의
#             popup_text = f"유형: {row['유형']}<br>사고 일자: {row['연월일']}<br>위치: {row['위도_변환']}, {row['경도_변환']}<br>사고장소: {row['위치']}"
#             folium.CircleMarker(
#                 location=(row['위도_변환'], row['경도_변환']),
#                 radius=3,
#                 color=accident_color,
#                 fill=True,
#                 fill_color=accident_color,
#                 fill_opacity=1.0,  # 내부 채움 불투명도
#                 popup=popup_text,
#                 tooltip=tooltip_text  # 툴팁 추가
#             ).add_to(feature_group)
        
#         feature_group.add_to(m)  # FeatureGroup을 지도 객체에 추가
#         groups['사고 원인'].append(feature_group)  # 사고 원인별로 그룹에 FeatureGroup을 추가합니다.

#     # 사고 원인별 그룹을 그룹화된 레이어 컨트롤로 추가
#     GroupedLayerControl(groups=groups, collapsed=False, exclusive_groups=False).add_to(m)
    

    # 레이어 컨트롤 추가하여 사용자가 레이어 선택 가능하게 함
    folium.LayerControl().add_to(m)

    
    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:13px; left: 10px; bottom: 10px;'>
        
    <div class='legend-title'>범례</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
        <li><span style='background:blue;opacity:1;'></span><strong>기존 AED 설치지점</strong></li>
        <li><span style='background:#DC3323;opacity:1;'></span><strong>심장사고지점</strong></li>
        <li><span style='color:#E05EC6; font-size: 14px;'>&#10084;</span><strong>AED 추가설치 예측지점</strong></li>

    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Add the legend to the map
    macro = MacroElement()
    macro._template = Template(template)
    macro.add_to(m)

    return m
















########################################### 핫스팟과 추락위험지역이 입력한 거리보다 넘어간 핫스팟 위치 표시 ##################################
def make_hotspot_fall(selected_national_park_accident,selected_npark_boundary,df_fall,distance=500):
    single_polygon = selected_npark_boundary['geometry'].unary_union
    # GeoJSON 형식으로 변환
    geojson_polygon = single_polygon.__geo_interface__
    resolution=9
    # 북한산 국립공원을 커버하는 H3 육각형 인덱스 생성
    hexagons = list(h3.polyfill_geojson(geojson_polygon, resolution))

    # polygon = seoul_npark_boundary.iloc[0]['geometry']
    # 육각형 경계 좌표 생성
    hexagon_polygons = [Polygon(h3.h3_to_geo_boundary(hex, geo_json=True)) for hex in hexagons]

    # GeoDataFrame 생성
    gdf_polygon = gpd.GeoDataFrame(geometry=hexagon_polygons)

    # 심장문제만 필터링
    selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['유형']=='추락']
    prices = gpd.sjoin(gdf_polygon, selected_national_park_accident[['geometry']], op='contains')

    # 각 geometry에 대한 카운트를 계산
    nbr_avg_price = prices.groupby('geometry').size().reset_index(name='acc_counts')
    # 병합하기
    nbr_final = gdf_polygon.merge(nbr_avg_price, on='geometry', how='left')
    # 병합 결과에서 결측값을 0으로 채우기
    nbr_final['acc_counts'] = nbr_final['acc_counts'].fillna(0)
    w =  lps.weights.Queen.from_dataframe(nbr_final)
    w.transform = 'r'

    # 속성 유사성(Attribute similarity) / 가중치 적용 사고발생건수 추가
    nbr_final['weighted_acc_counts'] = lps.weights.lag_spatial(w, nbr_final['acc_counts'])

    # 광역적 공간 자기상관 / 상관계수값 출력 
    y = nbr_final.acc_counts
    moran = esda.Moran(y, w)
    moran_local = Moran_Local(y, w)
    nbr_final['Cluster'] = moran_local.q
    cluster_map = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    nbr_final['Cluster_Label'] = nbr_final['Cluster'].map(cluster_map)
    moran_local = Moran_Local(y, w)
    # p-value 기준으로 유의미한 결과만 필터링
    p_threshold = 0.05  # 유의수준 설정
    nbr_final['Significant'] = moran_local.p_sim < p_threshold

    # 유의미하지 않은 관측치에 대해 'NS' 라벨 할당
    nbr_final['Cluster_Label'] = nbr_final.apply(lambda row: 'NS' if not row['Significant'] else row['Cluster_Label'], axis=1)

    # 'Significant' 열은 이제 필요 없으므로 제거하거나, 유지하려면 이 단계를 생략
    nbr_final.drop(columns=['Significant'], inplace=True)

    center = nbr_final.geometry.centroid.unary_union.centroid
    m = folium.Map(location=[center.y, center.x], zoom_start=12)
    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)

    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)

    # 클러스터 레이어 설정
    cluster_colors_핫스팟 = {
        'HH': 'red',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'transparent',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function_핫스팟(feature):
        return {
            'fillColor': cluster_colors_핫스팟.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"

    # 클러스터 레이어 설정
    cluster_colors_콜드스팟 = {
        'HH': 'transparent',
        'HL': 'transparent',
        'LH': 'transparent',
        'LL': 'blue',
        'NS': 'transparent'
    }

    # 클러스터에 따른 스타일 설정 함수
    def style_function_콜드스팟(feature):
        return {
            'fillColor': cluster_colors_콜드스팟.get(feature['properties']['Cluster_Label'], 'gray'),
            'color': 'transparent',
            'weight': 1,
            'fillOpacity': 0.5
        }
    nbr_final.crs = "EPSG:4326"



    # 클러스터 레이어 추가
    cluster_layer_핫스팟 = folium.FeatureGroup(name='추락사고 핫스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function_핫스팟,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['추락사고수:']
        )
    ).add_to(cluster_layer_핫스팟)
    cluster_layer_핫스팟.add_to(m)


    # 클러스터 레이어 추가
    cluster_layer_콜드스팟 = folium.FeatureGroup(name='추락사고 콜드스팟')
    folium.GeoJson(
        nbr_final,
        style_function=style_function_콜드스팟,
        tooltip=folium.GeoJsonTooltip(
            fields=['acc_counts'],
            aliases=['추락사고수:']
        )
    ).add_to(cluster_layer_콜드스팟)
    cluster_layer_콜드스팟.add_to(m)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # # 위치표지판 레이어 설정 및 추가
    # sign_layer = folium.FeatureGroup(name='다목적위치표지판')
    # for idx, row in sign_place.iterrows():
    #     folium.CircleMarker(
    #         location=(row['위도'], row['경도']),
    #         popup=row['위치'],
    #         radius=3,
    #         color='orange',
    #         fill=True,
    #         fill_color='orange',
    #         fill_opacity=0.8
    #     ).add_to(sign_layer)
    # sign_layer.add_to(m)




    # 추락위험지역 설정 및 추가
    fall_spot_layer = folium.FeatureGroup(name='기존 추락위험지역')
    for idx, row in df_fall.iterrows():
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=row['세부위치'],
            radius=3,
            color='blue',
            fill=True,
            fill_color='blue',
            fill_opacity=1
        ).add_to(fall_spot_layer)
    fall_spot_layer.add_to(m)

    # 추락사 사고지점 추가
    seoul_accident_fall_layer = folium.FeatureGroup(name='추락사고지점')
    for idx, row in selected_national_park_accident.iterrows():
            tooltip_text = f"유형: {row['유형']}<br>사고 일자: {row['연월일']}<br>위치: {row['위도_변환']}, {row['경도_변환']}<br>사고장소: {row['위치']}<br>고도: {float(row['고도']):.2f}<br>경사도: {float(row['경사도']):.2f}"                
            folium.CircleMarker(
            location=(row['위도_변환'], row['경도_변환']),
            popup=row['사고장소'],
            radius=3,
            color='red',
            fill=True,
            fill_color='red',
            fill_opacity=1,
            tooltip=tooltip_text
        ).add_to(seoul_accident_fall_layer)
    seoul_accident_fall_layer.add_to(m)



    def filter_hotspots_far_from_fall(nbr_final, df_fall, min_distance=100, cluster_label='HH'):
        # 안전쉼터 데이터를 GeoDataFrame으로 변환
        df_fall_gdf = gpd.GeoDataFrame(
            df_fall,
            geometry=gpd.points_from_xy(df_fall.경도, df_fall.위도),
            crs='EPSG:4326'
        )

        # 좌표계 변환
        nbr_final_utm = nbr_final.to_crs(epsg=5174)
        df_fall_utm = df_fall_gdf.to_crs(epsg=5174)

        # 'HH' 클러스터 라벨이 지정된 핫스팟 선택
        nbr_final_핫스팟 = nbr_final_utm[nbr_final_utm['Cluster_Label'] == cluster_label]

        # nbr_final_핫스팟이 비어있는 경우 예외 발생
        if len(nbr_final_핫스팟) == 0:
            raise ValueError("No hotspots found with the specified cluster label.")

        # 각 사고 핫스팟에서 가장 가까운 안전쉼터까지의 최소 거리 계산
        def calculate_min_distance_to_fall(hotspot, shelters):
            # 사고 핫스팟과 모든 안전쉼터 사이의 거리 계산 후 최소값 반환
            distances = shelters.geometry.distance(hotspot.geometry)
            return distances.min()
        # 최소 거리 계산
        nbr_final_핫스팟['min_distance_to_shelter'] = nbr_final_핫스팟.apply(
            lambda hotspot: calculate_min_distance_to_fall(hotspot, df_fall_utm), axis=1
        )

        # 지정된 최소 거리 이상 떨어진 핫스팟 필터링
        hotspots_far_from_shelters = nbr_final_핫스팟[nbr_final_핫스팟['min_distance_to_shelter'] > min_distance]

        # WGS84 좌표계로 변환
        hotspots_far_from_shelters_wgs84 = hotspots_far_from_shelters.to_crs(epsg=4326)
        
        return hotspots_far_from_shelters_wgs84

    ########## 이격거리 조절가능 ########################## 
    out_hotspot_layer = folium.FeatureGroup(name='추락위험지역 예측지점')
    try:
        for idx, row in filter_hotspots_far_from_fall(nbr_final, df_fall, distance, 'HH').iterrows():
            folium.Marker(
                location=[row.geometry.centroid.y, row.geometry.centroid.x],
                icon=folium.Icon(icon="arrow-down"),
            ).add_to(out_hotspot_layer)
        out_hotspot_layer.add_to(m)
    except ValueError as e:
        st.error("추락사고 발생 건수가 적어 지도 분석이 어려워요. 다른 공원을 분석해주세요.")

#        # 사고 원인별 색상 사전 정의
#     palette = sns.color_palette('bright')

#     # 사건 유형에 대한 색상 딕셔너리
#     color_dict = {
#         '실족ㆍ골절': palette[0],
#         '기타': palette[1],
#         '일시적고립': palette[2],
#         '탈진경련': palette[3],
#         '낙석ㆍ낙빙': palette[4],
#         '추락': palette[5],
#         '심장사고': palette[6],
#         '해충피해': palette[7],
#         '익수': palette[8],
#     }

#     # 컬러 팔레트에 해당하는 RGB 값을 hex 코드로 변환
#     color_dict_hex = {key: mcolors.rgb2hex(value) for key, value in color_dict.items()}
#     # 사고 원인별로 레이어 그룹 생성 및 추가
#     # 사고 원인별로 레이어 그룹 생성 및 추가
#     accident_types = selected_national_park_accident['유형'].unique()
#    # 사고 원인별로 레이어 그룹 생성 및 추가
#     groups = {'사고 원인': []}  # 사고 원인별 그룹을 담을 리스트를 생성합니다.

#     for i, color in color_dict_hex.items():
#         type_accident = selected_national_park_accident[selected_national_park_accident['유형'] == i]
#         accident_color = color  # 사고 원인별로 정의된 색상 사용
#         feature_group = folium.FeatureGroup(name=i)  # 사고 원인별 FeatureGroup 생성

#         # 사고 위치에 대한 CircleMarker 추가 및 툴팁 정보 설정
#         for idx, row in type_accident.iterrows():
#             tooltip_text = f"유형: {row['유형']}<br>사고 일자: {row['연월일']}"  # 툴팁 텍스트 정의
#             popup_text = f"유형: {row['유형']}<br>사고 일자: {row['연월일']}<br>위치: {row['위도_변환']}, {row['경도_변환']}<br>사고장소: {row['위치']}"
#             folium.CircleMarker(
#                 location=(row['위도_변환'], row['경도_변환']),
#                 radius=3,
#                 color=accident_color,
#                 fill=True,
#                 fill_color=accident_color,
#                 fill_opacity=1.0,  # 내부 채움 불투명도
#                 popup=popup_text,
#                 tooltip=tooltip_text  # 툴팁 추가
#             ).add_to(feature_group)
        
#         feature_group.add_to(m)  # FeatureGroup을 지도 객체에 추가
#         groups['사고 원인'].append(feature_group)  # 사고 원인별로 그룹에 FeatureGroup을 추가합니다.

#     # 사고 원인별 그룹을 그룹화된 레이어 컨트롤로 추가
#     GroupedLayerControl(groups=groups, collapsed=False, exclusive_groups=False).add_to(m)
    
    # 레이어 컨트롤 추가하여 사용자가 레이어 선택 가능하게 함
    folium.LayerControl().add_to(m)

    template = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" integrity="sha384-ekzFgdUyJibDJSR+Jb+wdX4Axq7rKjHcXdMI1a3iFxoZou1GEGq3w5LaD1aMhohc" crossorigin="anonymous">
    </head>
    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:13px; left: 10px; bottom: 10px;'>
        
    <div class='legend-title'>범례</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
        <li><span style='background:blue;opacity:1;'></span><strong>기존 추락위험지역</strong></li>
        <li><span style='background:#DC3323;opacity:1;'></span><strong>추락사고지점</strong></li>
        <li><span style='font-size: 15px;'><i class="fa-solid fa-circle-down" style="color: #599CD5;"></i></span><strong>추락위험지역 예측지점</strong></li>
    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Add the legend to the map
    macro = MacroElement()
    macro._template = Template(template)
    macro.add_to(m)

    return m









# Heat map
def make_heatmap(selected_national_park_accident,selected_npark_boundary):
    m = v_world(selected_national_park_accident)
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                        title='전체 화면',     # 마우스 오버시 표시될 텍스트
                        title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                        force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)

     # seoul_npark_boundary GeoDataFrame을 GeoJson으로 변환 및 추가
    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)
    
    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)



    # 사고 위치 데이터 준비 (위도, 경도)
    accident_locations = selected_national_park_accident[['위도_변환', '경도_변환']].values.tolist()

    # 히트맵 레이어 생성
    heat_map = plugins.HeatMap(accident_locations, radius=15, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'})

    # 레이어 그룹 생성 및 히트맵 레이어 추가
    layer_group = folium.FeatureGroup().add_child(heat_map)

    # 레이어 그룹을 지도 객체에 추가
    m.add_child(layer_group)

    # 레이어 컨트롤 추가
    folium.LayerControl().add_to(m)

    return m

def plot_donut_chart(df):
    # value_counts를 사용해 각 카테고리의 빈도수 계산
    value_counts = df['유형'].value_counts()

    # 각 카테고리에 대한 색상을 지정합니다.
    colors = {
        '실족ㆍ골절': '#055AFF',
        '기타': '#FC9238',
        '일시적고립': '#75D456',
        '탈진경련': '#C82E23',
        '낙석ㆍ낙빙': '#8E44AD',
        '추락': '#784315',
        '심장사고': '#E05EC6',
        '해충피해': 'gray',
        '익수': '#F4D03F'
    }

    # 각 카테고리에 대한 색상 리스트 생성
    category_colors = [colors[category] for category in value_counts.index]

    # 도넛 차트를 위한 데이터와 레이아웃 설정
    fig = go.Figure(data=[go.Pie(labels=value_counts.index, values=value_counts, hole=.3, marker=dict(colors=category_colors))])
    
    # 차트 제목 및 레이아웃 설정
    fig.update_layout(
        title_text='사고 원인 분포도',
        # 글씨 크기 조정
        title_font_size=20,
        legend_title_font_size=12,
        font=dict(size=18)
    )


    return fig

import geopandas as gpd
from shapely.geometry import Point, Polygon
from folium.plugins import MeasureControl
from folium import Marker, LayerControl, GeoJson,Icon, FeatureGroup
import folium

# 등급별 색상 지정
def color_producer(val):
    if val == '6등급':
        return 'lightgreen'
    elif val == '5등급':
        return 'green'
    elif val == '4등급':
        return 'yellow'
    elif val == '3등급':
        return 'orange'
    elif val == '2등급':
        return 'salmon'
    else:
        return 'red'
    
def fall_model(fall_gdf):

    map_center_lat, map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)

    first = fall_gdf[fall_gdf['위험도'] == '1등급']
    second = fall_gdf[fall_gdf['위험도'] == '2등급']
    third = fall_gdf[fall_gdf['위험도'] == '3등급']
    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12)

    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)
    # 국립공원 경계
    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)
    
    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)
    trail_layer.add_to(m)

    # GeoJson 레이어 추가
    folium.GeoJson(
        first,
        name="위험도 1등급",
        style_function=lambda feature: {
            'fillColor': color_producer(feature['properties']['위험도']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        highlight_function=lambda feature: {'weight': 3, 'color': 'blue'},
        tooltip=folium.GeoJsonTooltip(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도'],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # Tooltip 텍스트
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도', ],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # 팝업 텍스트
            localize=True
        )
    ).add_to(m)

    # GeoJson 레이어 추가
    folium.GeoJson(
        second,
        name="위험도 2등급",
        style_function=lambda feature: {
            'fillColor': color_producer(feature['properties']['위험도']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.4
        },
        highlight_function=lambda feature: {'weight': 3, 'color': 'blue'},
        tooltip=folium.GeoJsonTooltip(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도'],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # Tooltip 텍스트
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도', ],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # 팝업 텍스트
            localize=True
        )
    ).add_to(m)

    # GeoJson 레이어 추가
    folium.GeoJson(
        third,
        name="위험도 3등급",
        style_function=lambda feature: {
            'fillColor': color_producer(feature['properties']['위험도']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.4
        },
        highlight_function=lambda feature: {'weight': 3, 'color': 'blue'},
        tooltip=folium.GeoJsonTooltip(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도'],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # Tooltip 텍스트
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도', ],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # 팝업 텍스트
            localize=True
        )
    ).add_to(m)

    # 추락위험지역 설정 및 추가
    fall_spot_layer = folium.FeatureGroup(name='기존 추락위험지역')
    for idx, row in df_fall.iterrows():
        # 팝업을 위한 HTML 컨텐츠 생성
        popup_html = f"""
        <div style='font-family: "Arial", sans-serif; font-size: 12px;'>
            <h4>추락위험지역</h4>
            <p>세부위치: {row['세부위치']}<br>
            위도: {row['위도']:.5f}<br>
            경도: {row['경도']:.5f}</p>
        </div>
        """
        popup = folium.Popup(popup_html, max_width=250)
        
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            popup=popup,
            radius=3,
            color='purple',
            fill=True,
            fill_color='purple',
            fill_opacity=1
        ).add_to(fall_spot_layer)
    fall_spot_layer.add_to(m)

    # 안전쉼터 위치를 Marker로 추가
    shelter_layer = folium.FeatureGroup(name='안전쉼터')
    for idx, row in shelter.iterrows():
        # 팝업을 위한 HTML 컨텐츠 생성
        popup_html = f"""
        <div style='font-family: "Arial", sans-serif; font-size: 12px;'>
            <h4>{row['쉼터명']}</h4>
            <p>위도: {row['geometry'].y:.5f}<br>
            경도: {row['geometry'].x:.5f}</p>
        </div>
        """
        popup = folium.Popup(popup_html, max_width=250)

        # "대피소"가 쉼터명에 포함되어 있으면 마커 색을 'black'으로 설정
        icon_color = 'orange' if '대피소' in row['쉼터명'] else 'blue'

        # 마커 생성 (Font Awesome 아이콘 사용)
        Marker(
            location=[row['geometry'].y, row['geometry'].x],  # GeoDataFrame에서 위도, 경도 추출
            icon=Icon(icon='person-shelter', prefix='fa', color=icon_color),  # 조건에 따라 색상 변경
            popup=popup
        ).add_to(shelter_layer)

    shelter_layer.add_to(m)

    # 1000m 이상 거리에 있는 고위험 지역 필터링
    need_shelter_df = first[first['nearest_shelter_dist'] > resolution]
    need_shelter_df = need_shelter_df.to_crs(epsg=4326)

    # 설치 지역 레이어 추가
    need_shelter_layer = folium.FeatureGroup(name='설치 시급 지점')
    for idx, row in need_shelter_df.iterrows():
        # 팝업 컨텐츠 정의
        popup_content = f"""
        위도: {row['formatted_latitude']}<br>
        경도: {row['formatted_longitude']}<br>
        가까운 쉼터와의 거리: {row['nearest_shelter_dist']:.2f}m
        """
        popup = folium.Popup(popup_content, max_width=250)

        # 마커 추가
        Marker(
            location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
            popup=popup,
            icon=Icon(color='red', icon='exclamation-triangle', prefix='fa')  # FontAwesome 아이콘 사용
        ).add_to(need_shelter_layer)

    need_shelter_layer.add_to(m)
    # 점 찍고 점과 점 사이 거리 재기
    m.add_child(MeasureControl())

    # 모든 지점 위경도 표시
    m.add_child(
        folium.LatLngPopup()
    )

    # 레이어 켜고 끄는 컨트롤 추가
    LayerControl().add_to(m)
    
    template4 = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" integrity="sha384-ekzFgdUyJibDJSR+Jb+wdX4Axq7rKjHcXdMI1a3iFxoZou1GEGq3w5LaD1aMhohc" crossorigin="anonymous">
    </head>
    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:13px; left: 10px; bottom: 10px;'>
        
    <div class='legend-title'>위험도 범례</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
        <li><span style='background:red;opacity:1;'></span><strong>1등급 : 사고발생확률 95% 이상</strong></li>
        <li><span style='background:salmon;opacity:1;'></span><strong>2등급 : 사고발생확률 90% 이상</strong></li>
        <li><span style='background:orange;opacity:1;'></span><strong>3등급 : 사고발생확률 70% 이상</strong></li>
    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Add the legend to the map
    macro = MacroElement()
    macro._template = Template(template4)
    macro.add_to(m)
    
    return m

def heart_model(heart_gdf):

    map_center_lat, map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)

    first = heart_gdf[heart_gdf['위험도'] == '1등급']
    second = heart_gdf[heart_gdf['위험도'] == '2등급']
    third = heart_gdf[heart_gdf['위험도'] == '3등급']

    m = folium.Map(location=[map_center_lat, map_center_lon], zoom_start=12)

    # 전체 화면 버튼 추가
    fullscreen = Fullscreen(position='topleft',  # 버튼 위치
                            title='전체 화면',     # 마우스 오버시 표시될 텍스트
                            title_cancel='전체 화면 해제',  # 전체 화면 모드 해제 버튼의 텍스트
                            force_separate_button=True)  # 전체 화면 버튼을 별도의 버튼으로 표시
    m.add_child(fullscreen)
    # VWorld Satellite Layer 추가
    vworld_key = "BF677CB9-D1EA-3831-B328-084A9AE3CDCC"
    satellite_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Satellite/{{z}}/{{y}}/{{x}}.jpeg",
        attr='VWorld Satellite', 
        name='위성지도'
    ).add_to(m)

    # VWorld Hybrid Layer 추가
    hybrid_layer = folium.TileLayer(
        tiles=f"http://api.vworld.kr/req/wmts/1.0.0/{vworld_key}/Hybrid/{{z}}/{{y}}/{{x}}.png",
        attr='VWorld Hybrid', 
        name='지명표시', 
        overlay=True
    ).add_to(m)
    # 국립공원 경계
    geojson_data = json.loads(selected_npark_boundary.to_json())
    folium.GeoJson(
        geojson_data,
        name='국립공원 경계',
        style_function=lambda feature: {
            'color': 'yellow',
            'weight': 2,
            'fillOpacity': 0
        }
    ).add_to(m)

    # 탐방로 레이어 설정 및 추가
    trail_layer = folium.FeatureGroup(name='탐방로')
    folium.GeoJson(
        df_탐방로[df_탐방로.geometry.length > 0.001],
        name='Trails',
        style_function=lambda feature: {
            'fillColor': 'blue',
            'color': 'blue',
            'weight': 3,
            'fillOpacity': 0.4,
            'opacity': 0.4
        }
    ).add_to(trail_layer)

    trail_layer.add_to(m)
    
    # GeoJson 레이어 추가
    folium.GeoJson(
        first,
        name="위험도 1등급",
        style_function=lambda feature: {
            'fillColor': color_producer(feature['properties']['위험도']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.6
        },
        highlight_function=lambda feature: {'weight': 3, 'color': 'blue'},
        tooltip=folium.GeoJsonTooltip(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도'],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # Tooltip 텍스트
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도', ],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # 팝업 텍스트
            localize=True
        )
    ).add_to(m)

    # GeoJson 레이어 추가
    folium.GeoJson(
        second,
        name="위험도 2등급",
        style_function=lambda feature: {
            'fillColor': color_producer(feature['properties']['위험도']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.4
        },
        highlight_function=lambda feature: {'weight': 3, 'color': 'blue'},
        tooltip=folium.GeoJsonTooltip(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도'],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # Tooltip 텍스트
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도', ],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # 팝업 텍스트
            localize=True
        )
    ).add_to(m)

    # GeoJson 레이어 추가
    folium.GeoJson(
        third,
        name="위험도 3등급",
        style_function=lambda feature: {
            'fillColor': color_producer(feature['properties']['위험도']),
            'color': 'black',
            'weight': 1,
            'fillOpacity': 0.4
        },
        highlight_function=lambda feature: {'weight': 3, 'color': 'blue'},
        tooltip=folium.GeoJsonTooltip(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도'],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # Tooltip 텍스트
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=['formatted_latitude', 'formatted_longitude', '위험도', '고도', '경사도', ],
            aliases=['위도:', '경도:', '위험도:', '고도:', '경사도:'],  # 팝업 텍스트
            localize=True
        )
    ).add_to(m)

    # 스마트워치 레이어
    smartwatch_layer = folium.FeatureGroup(name='스마트워치')
    Q1 = park_watch['심박수'].quantile(0.25)
    Q3 = park_watch['심박수'].quantile(0.75)
    IQR = Q3 - Q1
    outlier_threshold = Q3 + 1.5 * IQR
    high_heart_rate_threshold = park_watch['심박수'].quantile(0.75)

    for idx, row in park_watch.iterrows():
        if row['심박수'] > outlier_threshold:
            color = '#664724'
        elif row['심박수'] >= high_heart_rate_threshold:
            color = '#BC8344'
        else:
            color = '#E1C6A9'
        folium.CircleMarker(
            location=(row['위도'], row['경도']),
            radius=3,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.5,
            opacity=0.5,
            tooltip=f"심박수: {row['심박수']}"
        ).add_to(smartwatch_layer)

    # 히트맵 레이어 생성
    heat_data = [[row['위도'], row['경도']] for index, row in park_watch.iterrows()]
    heat_layer = plugins.HeatMap(heat_data, radius=15, gradient={0.2: 'blue', 0.4: 'green', 0.6: 'yellow', 0.8: 'orange', 1: 'red'}, name='스마트워치 히트맵')

    # 안전쉼터 위치를 Marker로 추가
    shelter_layer = folium.FeatureGroup(name='안전쉼터')
    for idx, row in shelter.iterrows():
        # 팝업을 위한 HTML 컨텐츠 생성
        popup_html = f"""
        <div style='font-family: "Arial", sans-serif; font-size: 12px;'>
            <h4>{row['쉼터명']}</h4>
            <p>위도: {row['geometry'].y:.5f}<br>
            경도: {row['geometry'].x:.5f}</p>
        </div>
        """
        popup = folium.Popup(popup_html, max_width=250)

        # "대피소"가 쉼터명에 포함되어 있으면 마커 색을 'black'으로 설정
        icon_color = 'orange' if '대피소' in row['쉼터명'] else 'blue'

        # 마커 생성 (Font Awesome 아이콘 사용)
        Marker(
            location=[row['geometry'].y, row['geometry'].x],  # GeoDataFrame에서 위도, 경도 추출
            icon=Icon(icon='person-shelter', prefix='fa', color=icon_color),  # 조건에 따라 색상 변경
            popup=popup
        ).add_to(shelter_layer)

    # 안전쉼터 위치를 Marker로 추가
    aed_layer = folium.FeatureGroup(name='AED')
    for idx, row in aed.iterrows():
        # 팝업을 위한 HTML 컨텐츠 생성
        popup_html = f"""
        <div style='font-family: "Arial", sans-serif; font-size: 12px;'>
            <h4>{row['명칭']}</h4>
            <p>위도: {row['geometry'].y:.5f}<br>
            경도: {row['geometry'].x:.5f}</p>
        </div>
        """
        popup = folium.Popup(popup_html, max_width=250)
        
        # 마커 생성 (Font Awesome 아이콘 사용)
        Marker(
            location=[row['geometry'].y, row['geometry'].x],  # GeoDataFrame에서 위도, 경도 추출
            icon=Icon(icon='heart-circle-check', prefix='fa', color='pink', icon_color = 'black'),  # Font Awesome 아이콘 설정
            popup=popup
        ).add_to(aed_layer)

    # 1000m 이상 거리에 있는 고위험 지역 필터링
    need_shelter_df = first[first['nearest_shelter_dist'] > resolution]
    need_shelter_df = need_shelter_df.to_crs(epsg=4326)

    # 설치 지역 레이어 추가
    need_shelter_layer = folium.FeatureGroup(name='설치 시급 지점')
    for idx, row in need_shelter_df.iterrows():
        # 팝업 컨텐츠 정의
        popup_content = f"""
        위도: {row['formatted_latitude']}<br>
        경도: {row['formatted_longitude']}<br>
        가까운 쉼터와의 거리: {row['nearest_shelter_dist']:.2f}m
        """
        popup = folium.Popup(popup_content, max_width=250)

        # 마커 추가
        Marker(
            location=[row['geometry'].centroid.y, row['geometry'].centroid.x],
            popup=popup,
            icon=Icon(color='red', icon='exclamation-triangle', prefix='fa')  # FontAwesome 아이콘 사용
        ).add_to(need_shelter_layer)

    # 모든 레이어 추가
    shelter_layer.add_to(m)
    aed_layer.add_to(m)
    need_shelter_layer.add_to(m)
    smartwatch_layer.add_to(m)
    trail_layer.add_to(m)
    smartwatch_layer.add_to(m)
    heat_layer.add_to(m)

    # 점 찍고 점과 점 사이 거리 재기
    m.add_child(MeasureControl())

    # 모든 지점 위경도 표시
    m.add_child(
        folium.LatLngPopup()
    )

    # 레이어 켜고 끄는 컨트롤 추가
    LayerControl().add_to(m)
    
    template5 = """
    {% macro html(this, kwargs) %}

    <!doctype html>
    <html lang="en">
    <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>jQuery UI Draggable - Default functionality</title>
    <link rel="stylesheet" href="//code.jquery.com/ui/1.12.1/themes/base/jquery-ui.css">

    <script src="https://code.jquery.com/jquery-1.12.4.js"></script>
    <script src="https://code.jquery.com/ui/1.12.1/jquery-ui.js"></script>
    
    <script>
    $( function() {
        $( "#maplegend" ).draggable({
                        start: function (event, ui) {
                            $(this).css({
                                right: "auto",
                                top: "auto",
                                bottom: "auto"
                            });
                        }
                    });
    });

    </script>
    </head>
    <body>

    
    <head>
        <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0-beta3/css/all.min.css" integrity="sha384-ekzFgdUyJibDJSR+Jb+wdX4Axq7rKjHcXdMI1a3iFxoZou1GEGq3w5LaD1aMhohc" crossorigin="anonymous">
    </head>
    
    <div id='maplegend' class='maplegend' 
        style='position: absolute; z-index:9999; border:2px grey; background-color:rgba(255, 255, 255, 0.8);
        border-radius:6px; padding: 10px; font-size:13px; left: 10px; bottom: 10px;'>
        
    <div class='legend-title'>위험도 범례</div>
    <div class='legend-scale'>
    <ul class='legend-labels'>
        <li><span style='background:red;opacity:1;'></span><strong>1등급 : 사고발생확률 95% 이상</strong></li>
        <li><span style='background:salmon;opacity:1;'></span><strong>2등급 : 사고발생확률 90% 이상</strong></li>
        <li><span style='background:orange;opacity:1;'></span><strong>3등급 : 사고발생확률 70% 이상</strong></li>
    </ul>
    </div>
    </div>
    
    </body>
    </html>

    <style type='text/css'>
    .maplegend .legend-title {
        text-align: left;
        margin-bottom: 5px;
        font-weight: bold;
        font-size: 90%;
        }
    .maplegend .legend-scale ul {
        margin: 0;
        margin-bottom: 5px;
        padding: 0;
        float: left;
        list-style: none;
        }
    .maplegend .legend-scale ul li {
        font-size: 80%;
        list-style: none;
        margin-left: 0;
        line-height: 18px;
        margin-bottom: 2px;
        }
    .maplegend ul.legend-labels li span {
        display: block;
        float: left;
        height: 16px;
        width: 30px;
        margin-right: 5px;
        margin-left: 0;
        border: 1px solid #999;
        }
    .maplegend .legend-source {
        font-size: 80%;
        color: #777;
        clear: both;
        }
    .maplegend a {
        color: #777;
        }
    </style>
    {% endmacro %}"""

    # Add the legend to the map
    macro = MacroElement()
    macro._template = Template(template5)
    macro.add_to(m)
    # Map을 저장하거나 표시

    return m

def load_data(park_name):
    if park_name == "북한산":
        shelter = gpd.read_file("./data/북한산 안전쉼터.geojson")
        aed = gpd.read_file("./data/북한산 AED.geojson")
        fall_gdf = gpd.read_file("./data/북한산 추락사고(100)_gdf.geojson")
        heart_gdf = gpd.read_file("./data/북한산 심장사고(100)_gdf.geojson")
        park_watch = smartwatch[smartwatch['국립공원'] == '북한산']
    elif park_name == "설악산":
        shelter = gpd.read_file("./data/설악산 안전쉼터.geojson")
        aed = gpd.read_file("./data/설악산 AED.geojson")
        fall_gdf = gpd.read_file("./data/설악산 추락사고(100)_gdf.geojson")
        heart_gdf = gpd.read_file("./data/설악산 심장사고(100)_gdf.geojson")
        park_watch = smartwatch[smartwatch['국립공원'] == '설악산']
    else:
        # 기본값 또는 오류 처리
        shelter = gpd.GeoDataFrame()
        aed = gpd.GeoDataFrame()
        fall_gdf = gpd.GeoDataFrame()
        heart_gdf = gpd.GeoDataFrame()
        park_watch = gpd.GeoDataFrame()
    return shelter, aed, fall_gdf, heart_gdf, park_watch



#######################
# Sidebar
with st.sidebar:
    st.title('국립공원 Dashboard')
    nationpark_list = ['북한산', '설악산', '지리산', '무등산', '덕유산', '계룡산', '월출산', '태백산', '월악산', '내장산',
       '속리산', '주왕산', '소백산', '변산반도', '치악산', '오대산', '가야산', '다도해해상', '한려해상', '경주',
       '태안해안']
    st.selectbox('국립공원 선택', nationpark_list,key='selected_national_park')
    year_list = [2014, 2015, 2016, 2017, 2018, 2019, 2020, 2021, 2022, 2023]

    year_list.insert(0,'전체')
    year = st.multiselect('연도 선택',year_list,key='year',default='전체')
    gender_list = ['전체','남','여']
    st.selectbox('성별 선택',gender_list,key='gender')
    month = st.multiselect('월별 선택',
    ['전체','1월', '2월','3월','4월','5월','6월','7월','8월','9월','10월','11월','12월'],key='month',default='전체')
    age = st.multiselect('연령대 선택',
    ['전체','20대미만','20대', '30대','40대','50대', '60대', '70대 이상', '미상', '집단'],key='age',default='전체')
    resolution = st.slider('기존 안전시설물과 사고 핫스팟 중심점 간의 거리(m) 설정', 500, 3000, 500,100,key='distance')
    st.write('핫스팟 중심점과의 이격거리(m)를 초과한 지점이 곧 핫스팟 내 안전시설물 우선설치 필요 지점 예측을 말해요.')
    button = st.button('분석 시작')
    image1 = './logo/국공.svg'
    image2 = './logo/Bigleader.png'
    st.write('')
    st.write('')
    st.image(image1)
    st.image(image2)


# Dashboard Main Panel
if not button:
    # custom_html = """
    # <div class="banner">
    #     <img src="" alt="Banner Image">
    # </div>
    # <style>
    #     .banner {
    #         width: 160%;
    #         height: 200px;
    #         overflow: hidden;
    #     }
    #     .banner img {
    #         width: 100%;
    #         object-fit: cover;
    #     }
    # </style>
    # """
    # # Display the custom HTML
    # st.components.v1.html(custom_html)
    st.markdown("""
    <style>
    body {
    background-size: cover;
    background-attachment: fixed; /* 배경 이미지 고정 */
    }

    .stApp { /* Streamlit 앱의 최대 너비 조정 */
        margin: auto;
    }

    .title {
        font-size: 48px; /* 제목 크기 조정 */
        font-weight: bold;
        text-align: center;
        margin-top: 30px;
        color: #6A877F; /* 부제목 색상 */

    }
    .subtitle {
        font-size: 28px; /* 부제목 크기 조정 */
        text-align: center;
        margin-bottom: 30px;
        color: #6A877F; /* 부제목 색상 */
 
    }
    .content {
        font-family: 'Noto Sans KR', sans-serif; /* 본문 글씨체 변경 */
        font-size: 30px; /* 본문 글씨 크기 변경 */
        padding: 20px;
        background-color: rgba(255,255,255,0.8); /* 본문 배경색 추가 및 투명도 조정 */
        border-radius: 15px; /* 본문 모서리 둥글게 */
        box-shadow: 0 4px 8px rgba(0,0,0,0.1); /* 본문에 그림자 효과 추가 */
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

    # 페이지 타이틀
    st.markdown('<h1 class="title">국립공원 안전사고 분석 리포트</h1>', unsafe_allow_html=True)

    # 페이지 부제목 및 소개
    st.markdown('<h2 class="subtitle">- 안전사고를 줄이기 위한 분석 및 대책 지원 -</h2>', unsafe_allow_html=True)

    st.markdown("""
    <div class="content">
        <p>국립공원 내 안전사고를 분석하고 효과적인 예방대책을 마련하는 페이지입니다. <br>
        좌측 사이드바에서는 사고 패턴을 파악할 수 있도록 다양한 시각화 도구와 분석 결과를 제공합니다.</p>
        <p>좌측 사이드바를 클릭하여 분석 자료를 확인해보세요.</p>
    </div>
    """, unsafe_allow_html=True)
   # CSS 스타일
if button:
    with st.spinner('분석 중입니다. 잠시만 기다려주세요!'):
        # npark_boundary = gpd.read_file('./data/Protected_areas_OECM_Republic_of_Korea_ver_2023.shp', encoding='cp949')
        # park_data = pd.read_csv('./data/240301_final_data_ver2.csv')
        # safety_place = pd.read_csv('./data/안전쉼터_final.csv')
        # sign_place = pd.read_excel('./data/북한산 다목적 위치표지판 현황.xlsx')
        # df_탐방로 = gpd.read_file('./data/국립공원시설_선형시설.shp')
        # #######################

        # gdf_park_data = gpd.GeoDataFrame(park_data, 
        #                             geometry=gpd.points_from_xy(park_data.경도_변환, park_data.위도_변환),
        #                             crs='epsg:4326'
        #                             )
        # gdf_safety_place = gpd.GeoDataFrame(safety_place, 
        #                             geometry=gpd.points_from_xy(safety_place.경도, safety_place.위도),
        #                             crs='epsg:4326'
        #                             )
        # gdf_sign_place = gpd.GeoDataFrame(sign_place, 
        #                             geometry=gpd.points_from_xy(sign_place.경도, sign_place.위도),
        #                             crs='epsg:4326'
        #                             )
        
        # GeoPackage 파일로부터 GeoDataFrame 불러오기
        plt.rcParams['font.family'] ='Malgun Gothic'
        plt.rcParams['axes.unicode_minus'] =False
        gdf_park_data = gpd.read_file("./data/park_data2.gpkg", layer='park_data2')
        sign_place = gpd.read_file("./data/sign_place.gpkg", layer='sign_place')
        df_탐방로 = gpd.read_file('./data/국립공원시설_선형시설.shp')
        npark_boundary = gpd.read_file('./data/npark_boundary.gpkg',layer='npark_boundary')
        df_AED = pd.read_csv('./data/AED_final.csv')
        df_fall = pd.read_csv('./data/추락위험지역_final.csv')
        safety_place = pd.read_csv("./data/안전쉼터_final.csv")
        smartwatch = gpd.read_file("./data/smartwatch_gdf.geojson")
        # bukhan_shelter = gpd.read_file("./data/북한산 안전쉼터.geojson")
        # bukhan_aed = gpd.read_file("./data/북한산 AED.geojson")
        # bukhan_fall_gdf = gpd.read_file("./data/북한산 추락사고_gdf.geojson")
        # bukhan_heart_gdf = gpd.read_file("./data/북한산 심장사고_gdf.geojson")
        # seorak_shelter = gpd.read_file("./data/설악산 안전쉼터.geojson")
        # seorak_aed = gpd.read_file("./data/설악산 AED.geojson")
        # seorak_fall_gdf = gpd.read_file("./data/설악산 추락사고_gdf.geojson")
        # seorak_heart_gdf = gpd.read_file("./data/설악산 심장사고_gdf.geojson")
        # #######################
        # npark_boundary = gpd.read_file('./data/npark_boundary.gpkg',layer='npark_boundary')
        # gdf_park_data = gpd.read_file("./data/park_data.gpkg", layer='park_data')
        # safety_place = gpd.read_file("./data/safety_place.gpkg", layer='safety_place')
        # sign_place = gpd.read_file("./data/sign_place.gpkg", layer='sign_place')
        
        selected_national_park = st.session_state['selected_national_park']
        safety_place = safety_place[safety_place['국립공원명']==selected_national_park]
        df_AED = df_AED[df_AED['국립공원명']==selected_national_park]
        df_fall = df_fall[df_fall['국립공원명']==selected_national_park]
        selected_npark_boundary = find_boundary(npark_boundary,selected_national_park)
        selected_npark_boundary_hotspot = find_boundary_hotspot(npark_boundary,selected_national_park)
        selected_national_park_accident = sjoin(gdf_park_data,selected_npark_boundary,selected_national_park)
        selected_national_park_accident_hotspot = sjoin(gdf_park_data,selected_npark_boundary_hotspot,selected_national_park)
        # 선택된 공원에 따른 데이터 로드
        shelter, aed, fall_gdf, heart_gdf, park_watch  = load_data(selected_national_park)
        if '전체' not in st.session_state['year']:
             selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['연도'].isin(st.session_state['year'])]
             selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['연도'].isin(st.session_state['year'])]
        if st.session_state['gender']!='전체':
            selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['성별']==st.session_state['gender']]
            selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['성별']==st.session_state['gender']]
        if '전체' not in st.session_state['month']:
             selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['월'].isin(st.session_state['month'])]
             selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['월'].isin(st.session_state['month'])]
        if '전체' not in st.session_state['age']:
            selected_national_park_accident = selected_national_park_accident[selected_national_park_accident['연령대'].isin(st.session_state['age'])]
            selected_national_park_accident_hotspot = selected_national_park_accident_hotspot[selected_national_park_accident_hotspot['연령대'].isin(st.session_state['age'])]
        # try:
        map_center_lat,map_center_lon = find_center_latitudes_longtitudes(selected_national_park_accident)
        
        vworld_key="BF677CB9-D1EA-3831-B328-084A9AE3CDCC" # VWorld API key
        layer = "Satellite" # VWorld layer
        tileType = "jpeg" # tile type
        accident_list = selected_national_park_accident['유형'].value_counts().index

        col = st.columns((2.5, 5.5), gap='medium')

        with col[0]:
            st.metric(label="사고 건수", value=len(selected_national_park_accident))
            fig1 = plot_donut_chart(selected_national_park_accident)
            st.plotly_chart(fig1, use_container_width=True)
            st.markdown("""
                    <div class="content">
                    <p><b>차트 활용법</b><br>
                    1. 좌측 사이드바를 닫고 확대된 그래프를 만나보세요.<br>
                    2. 차트를 클릭하여 사고건수를 확인해보세요.</p>
                    </div>
                    """, unsafe_allow_html=True)


        # with col[1]:
        #     st.markdown('#### 사고 현황판')
        #     tab1, tab2, tab3, tab4, tab5, tap6, tap7 = st.tabs(["사고 현황", "전체사고 히트맵","추락사고 예측","심장사고 예측", "추락위험지역 선정", '안전쉼터위치 선정', 'AED위치 선정'])
        #     with tab1:
        #         #col1 = st.columns([8.1, 1.9])
        #         #with col1[0]:
        #             # 지도 생성
        #         m,color_dict = make_pointplot(selected_national_park_accident,selected_npark_boundary)
        #         folium_static(m)

        #             # 이미지를 바이트로 변환
        #             #image = Image.open("./legend.png")  # 이미지 경로를 수정하세요
        #             #img_byte_arr = io.BytesIO()
        #             #image.save(img_byte_arr, format='PNG')
        #             #img_byte_arr = img_byte_arr.getvalue()
        #             #st.image(img_byte_arr, use_column_width=True)

        #             #st.markdown("""<div class="content">
        #             #<p> → 좌측 사이드바를 닫으시면 범례 정보를 더 큰 글씨로 만나보실 수 있어요.
        #             #</div>
        #             #""",unsafe_allow_html=True)

        #             #with col1[1]:
        #             # # 데이터 준비
        #             # df = pd.DataFrame(list(color_dict.items()), columns=['유형', '범례'])
        #             # # 색상을 나타내는 HTML 코드로 셀을 변환
        #             # df['범례'] = df['범례'].apply(lambda x: f'<div style="width: 26px; height: 20px; background-color: {x};"></div>')

        #             # # DataFrame을 HTML로 변환
        #             # html = df.to_html(escape=False,index=False)

        #             # # Streamlit에 HTML 표시
        #             # st.markdown(html, unsafe_allow_html=True)


            
        #     with tab2:
        #         # 지도 생성
        #         m2 = make_heatmap(selected_national_park_accident,selected_npark_boundary)
        #         folium_static(m2)
            
        #     with tab3:
        #         #col1 = st.columns([8.1, 1.9])
        #         #with col1[0]:
        #         # 지도 생성
        #         m3 = fall_model(fall_gdf)
        #         folium_static(m3)
        #         # with col1[1]:
        #         #     # Streamlit에 HTML 표시
        #         #     st.markdown(html, unsafe_allow_html=True)

        #     with tab4:
        #         #col1 = st.columns([8.1, 1.9])
        #         #with col1[0]:
        #         # 지도 생성
        #         m4 = heart_model(heart_gdf)
        #         folium_static(m4)
        #         # with col1[1]:
        #         #     # Streamlit에 HTML 표시
        #         #     st.markdown(html, unsafe_allow_html=True)

        #     with tab5:
        #         #col1 = st.columns([8.1, 1.9])
        #         #with col1[0]:
        #             # 지도 생성
        #         m5 = make_hotspot_fall(selected_national_park_accident_hotspot,selected_npark_boundary_hotspot,df_fall,st.session_state['distance'])
        #         folium_static(m5)
        #         # with col1[1]:
        #         #     # Streamlit에 HTML 표시
        #         #     st.markdown(html, unsafe_allow_html=True)
                
        #     with tap6:
        #         #col1 = st.columns([8.1, 1.9])
        #         #with col1[0]:
        #             # 지도 생성
        #         m6 = make_hotspot_safetyplace(selected_national_park_accident_hotspot,selected_npark_boundary_hotspot,safety_place,st.session_state['distance'])
        #         folium_static(m6)
        #             #st.image(img_byte_arr, use_column_width=True)
        #         # with col1[1]:
        #         #     # Streamlit에 HTML 표시
        #         #     st.markdown(html, unsafe_allow_html=True)   

        #     with tap7:
        #         #col1 = st.columns([8.1, 1.9])
        #         #with col1[0]:
        #             # 지도 생성
        #         m7 = make_hotspot_heart(selected_national_park_accident_hotspot,selected_npark_boundary_hotspot,df_AED,st.session_state['distance'])
        #         folium_static(m7)
        #         # with col1[1]:
        #         #     # Streamlit에 HTML 표시
        #         #     st.markdown(html, unsafe_allow_html=True) 
        
        with col[1]:
            st.markdown('#### 사고 현황판')

            if selected_national_park in ['북한산', '설악산']:
                tabs = st.tabs(["사고 현황", "전체사고 히트맵", "추락사고 예측", "심장사고 예측", "추락위험지역 선정", '안전쉼터위치 선정', 'AED위치 선정'])
            else:
                tabs = st.tabs(["사고 현황", "전체사고 히트맵", "추락위험지역 선정", '안전쉼터위치 선정', 'AED위치 선정'])

            with tabs[0]:
                # 지도 생성
                m, color_dict = make_pointplot(selected_national_park_accident, selected_npark_boundary)
                folium_static(m)

            with tabs[1]:
                # 지도 생성
                m2 = make_heatmap(selected_national_park_accident, selected_npark_boundary)
                folium_static(m2)
            tab_index = 2
            if selected_national_park in ['북한산', '설악산']:
                with tabs[tab_index]:
                    # 지도 생성
                    m3 = fall_model(fall_gdf)
                    folium_static(m3)
                    st.write("설치 시급지점이란?")
                    st.write("왼쪽 사이드바에서 설정한 이격거리로 사고위험확률 95% 이상인 구간 중심점과 모든 기존 안전쉼터와의 거리를 재고, 설정한 거리보다 더 멀리 떨어진 지점을 설치 시급지점으로 볼 수 있어요.")

                with tabs[tab_index + 1]:
                    # 지도 생성
                    m4 = heart_model(heart_gdf)
                    folium_static(m4)
                    st.write("설치 시급지점이란?")
                    st.write("왼쪽 사이드바에서 설정한 이격거리로 사고위험확률 95% 이상인 구간 중심점과 모든 기존 안전쉼터와의 거리를 재고, 설정한 거리보다 더 멀리 떨어진 지점을 설치 시급지점으로 볼 수 있어요.")
                
                tab_index += 2

            with tabs[tab_index]:
                # 지도 생성
                m5 = make_hotspot_fall(selected_national_park_accident_hotspot, selected_npark_boundary_hotspot, df_fall, st.session_state['distance'])
                folium_static(m5)

            with tabs[tab_index + 1]:
                # 지도 생성
                m6 = make_hotspot_safetyplace(selected_national_park_accident_hotspot, selected_npark_boundary_hotspot, safety_place, st.session_state['distance'])
                folium_static(m6)

            with tabs[tab_index + 2]:
                # 지도 생성
                m7 = make_hotspot_heart(selected_national_park_accident_hotspot, selected_npark_boundary_hotspot, df_AED, st.session_state['distance'])
                folium_static(m7)
