import streamlit as st

st.markdown("""
    <style>
    .fullwidth-title {
        width: 50vw !important;           /* tràn full màn hình */
        position: relative;
        left: 50%;
        right: 50%;
        margin-left: -25vw;                /* kéo ra ngoài block-container */
        margin-right: -50vw;
        text-align: center;
        font-size: 60px !important;
        font-weight: 900 !important;
        margin-top: 8vh;
        margin-bottom: 12vh;
        color: #0033CC !important;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="fullwidth-title">Dự Báo Bệnh Đái Tháo Đường Bằng Mô Hình Học Có Giám Sát</div>', unsafe_allow_html=True)


st.write("""
Đây là trang web giới thiệu và triển khai các mô hình học máy có giám sát để dự báo bệnh đái tháo đường (tiểu đường). 
Trang web cung cấp cái nhìn tổng quan về cách sử dụng các mô hình học có giám sát để phân loại và dự đoán bệnh dựa trên dữ liệu y tế.
""")

st.subheader("Các Nội Dung Chính Của Trang Web")
st.markdown("""
- **Decision Tree**: Giới thiệu và triển khai mô hình cây quyết định để dự báo bệnh đái tháo đường.
- **Random Forest**: Sử dụng mô hình rừng ngẫu nhiên để cải thiện độ chính xác dự báo.
- **Grid Search**: Tối ưu hóa siêu tham số cho các mô hình bằng phương pháp tìm kiếm lưới.
- **Cross-Validation (CV)**: Đánh giá mô hình bằng kỹ thuật kiểm chứng chéo để đảm bảo tính tổng quát hóa.
""")

st.info("Đi tới sidebar để chọn chức năng phân tích chi tiết.")