#include "TM1637.h"

unsigned char tab[] =
{
    0x3F,/*0*/
    0x06,/*1*/
    0x5B,/*2*/
    0x4F,/*3*/
    0x66,/*4*/
    0x6D,/*5*/
    0x7D,/*6*/
    0x07,/*7*/
    0x7F,/*8*/
    0x6F,/*9*/
    0x77,/*10 A*/
    0x7C,/*11 b*/
    0x58,/*12 c*/
    0x5E,/*13 d*/
    0x79,/*14 E*/
    0x71,/*15 F*/
    0x76,/*16 H*/
    0x38,/*17 L*/
    0x54,/*18 n*/
    0x73,/*19 P*/
    0x3E,/*20 U*/
    0x00,/*21 黑屏*/
};

// 最高位设置为1时显示 数码管上的":" 符号
unsigned char disp_num[] = {0x3F, 0x06 | 0x80, 0x5B, 0x4F, 0x66, 0x6D};			//存放6个数码管要显示的内容


//起始位 CLK为高电平时，DIO由高变低
void TM1637_Start( void )
{
    TM1637_DIO_OUT();
    TM1637_CLK_1;
    TM1637_DIO_1;
    HAL_Delay( 2 );
    TM1637_DIO_0;
}

//等待应答 传输数据正确时，在第八个时钟下降沿，芯片内部会产生一个ACK信号，将DIO管脚拉低，在第九个时钟结束之后释放DIO总线。
void TM1637_Ack( void )
{
    TM1637_DIO_IN();
    TM1637_CLK_0;
    HAL_Delay( 5 );													//在第八个时钟下降沿之后延时 5us，开始判断 ACK 信号
    while( HAL_GPIO_ReadPin(GPIOC, DIO_Pin) );								//等待应答位  这一行代码也可以不要 不影响实际使用效果 在使用软件仿真的时候需要屏蔽这句代码，否则程序就会卡在这里。
    TM1637_CLK_1;
    HAL_Delay( 2 );
    TM1637_CLK_0;
}

//停止位 CLK为高电平时，DIO由低变高
void TM1637_Stop( void )
{
    TM1637_DIO_OUT();
    TM1637_CLK_0;
    HAL_Delay( 2 );
    TM1637_DIO_0;
    HAL_Delay( 2 );
    TM1637_CLK_1;
    HAL_Delay( 2 );
    TM1637_DIO_1;
}
//输入数据在CLK的低电平变化，在CLK的高电平被传输。
//每传输一个字节，芯片内部在第八个时钟下降沿产生一个ACK
// 写一个字节
void TM1637_WriteByte( unsigned char oneByte )
{
    unsigned char i;
    TM1637_DIO_OUT();
    for( i = 0; i < 8; i++ )
    {
        TM1637_CLK_0;
        if( oneByte & 0x01 )									//低位在前
        {
            TM1637_DIO_1;
        }
        else
        {
            TM1637_DIO_0;
        }
        HAL_Delay( 3 );
        oneByte = oneByte >> 1;
        TM1637_CLK_1;
        HAL_Delay( 3 );
    }
}

//写显示寄存器  地址自增
void TM1637_Display_INC( void )
{
    unsigned char i;
    TM1637_Start();
    TM1637_WriteByte( 0x40 );									//写数据到显示寄存器 40H 地址自动加1 模式,44H 固定地址模式,本程序采用自加1模式
    TM1637_Ack();
    TM1637_Stop();
    TM1637_Start();
    TM1637_WriteByte( 0xC0 );									//地址命令设置 显示地址 00H
    TM1637_Ack();

    for( i = 0; i < 6; i++ )									//地址自加，不必每次都写地址
    {
        TM1637_WriteByte( disp_num[i] );			//发送数据   disp_num[]中存储6个数码管要显示的内容
        TM1637_Ack();
    }
    TM1637_Stop();
#if 0
    TM1637_Start();
    TM1637_WriteByte( 0x88 | 0x07 );					//开显示，最大亮度-----调节脉冲宽度控制0---7  脉冲宽度14/16
    TM1637_Ack();
    TM1637_Stop();
#endif
}

//写显示寄存器  地址不自增
// add 数码管的地址 0--5
// value 要显示的内容
void TM1637_Display_NoINC( unsigned char add, unsigned char value )
{
    TM1637_Start();
    TM1637_WriteByte( 0x44 );								//写数据到显示寄存器 40H 地址自动加1 模式,44H 固定地址模式,本程序采用自加1模式
    TM1637_Ack();
    TM1637_Stop();

    TM1637_Start();
    TM1637_WriteByte( 0xC0 | add );					//地址命令设置 显示地址 C0H---C5H
    TM1637_Ack();

    TM1637_WriteByte( value );			 				//发送数据   value存储要显示的内容
    TM1637_Ack();
    TM1637_Stop();
#if 0
    TM1637_Start();
    TM1637_WriteByte( 0x88 | 0x07 );							//开显示，最大亮度-----调节脉冲宽度控制0---7  脉冲宽度14/16
    TM1637_Ack();
    TM1637_Stop();
#endif
}

// level : 设置亮度等级  0---7
void TM1637_SetBrightness( unsigned char level )
{
    TM1637_Start();
    TM1637_WriteByte( 0x88 | level );							//开显示，最大亮度-----调节脉冲宽度控制0---7  脉冲宽度14/16
    TM1637_Ack();
    TM1637_Stop();
}

