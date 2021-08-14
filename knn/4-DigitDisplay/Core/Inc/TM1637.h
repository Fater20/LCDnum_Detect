#ifndef __TM1637_H
#define __TM1637_H

# include "main.h"
extern unsigned char tab[];
extern unsigned char disp_num[];

//IO方向设置		   								0011输出模式   1000上下拉输入模式
#define TM1637_DIO_IN()      {GPIOC->MODER&=0XFFFFFFFF;GPIOC->MODER|=(uint32_t)0<<4;}
#define TM1637_DIO_OUT()     {GPIOC->MODER&=0XFFFFFFFF;GPIOC->MODER|=(uint32_t)1<<4;}

#define TM1637_CLK_0 HAL_GPIO_WritePin(GPIOC, CLK_Pin, GPIO_PIN_RESET);
#define TM1637_CLK_1 HAL_GPIO_WritePin(GPIOC, CLK_Pin, GPIO_PIN_SET);
#define TM1637_DIO_0 HAL_GPIO_WritePin(GPIOC, DIO_Pin, GPIO_PIN_RESET);
#define TM1637_DIO_1 HAL_GPIO_WritePin(GPIOC, DIO_Pin, GPIO_PIN_SET);

void TM1637_Start( void );
void TM1637_Ack( void );
void TM1637_Stop( void );
void TM1637_WriteByte( unsigned char oneByte );
unsigned char TM1637_ScanKey( void );
void TM1637_NixieTubeDisplay( void );

void TM1637_SetBrightness( unsigned char level );
void TM1637_Display_INC( void );
void TM1637_Display_NoINC( unsigned char add, unsigned char value );
unsigned char TM1637_KeyProcess( void );

#endif
