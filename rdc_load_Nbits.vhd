----------------------------------------------------------------------------------
-- Company: 
-- Engineer: 
-- 
-- Create Date: 10/14/2021 09:51:10 AM
-- Design Name: 
-- Module Name: rdc_load_Nbits - Behavioral
-- Project Name: 
-- Target Devices: 
-- Tool Versions: 
-- Description: 
-- 
-- Dependencies: 
-- 
-- Revision:
-- Revision 0.01 - File Created
-- Additional Comments:
-- 
----------------------------------------------------------------------------------


library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

-- Uncomment the following library declaration if using
-- arithmetic functions with Signed or Unsigned values
--use IEEE.NUMERIC_STD.ALL;

-- Uncomment the following library declaration if instantiating
-- any Xilinx leaf cells in this code.
--library UNISIM;
--use UNISIM.VComponents.all;

entity rdc_load_Nbits is
    generic (N : integer := 10);
    Port ( RESET : in STD_LOGIC;
           CLK : in STD_LOGIC;
           ENABLE : in STD_LOGIC;
           MODE : in STD_LOGIC;
           INPUT : in STD_LOGIC;
           LOAD : in STD_LOGIC_VECTOR (N-1 downto 0);
           OUTPUT : out STD_LOGIC);
end rdc_load_Nbits;

architecture Behavioral of rdc_load_Nbits is

component OneBitRegister
    Port ( RESET : in STD_LOGIC;
       CLK : in STD_LOGIC;
       D : in STD_LOGIC;
       EN : in STD_LOGIC;
       Q : out STD_LOGIC);
end component;

component multiplexer
    Port ( MODE : in STD_LOGIC;
           LOAD : in STD_LOGIC;
           INPUT : in STD_LOGIC;
           OUTPUT : out STD_LOGIC);
end component;

signal sortieMultiplexers : STD_LOGIC_VECTOR (N-1 downto 0);
signal sortieRegistres : STD_LOGIC_VECTOR (N-1 downto 0);

begin

rdc_load: 
   for I in 0 to N-1 generate
      MULTIPLEXER0: if I=0 generate
         FIRST_MULTIPLEXER : multiplexer port map
              (MODE => MODE,
               LOAD => LOAD(I),
               INPUT => INPUT,
               OUTPUT => sortieMultiplexers(I));
       end generate MULTIPLEXER0;
      MULTIPLEXERX: if I>0 generate
        OTHER_MULTIPLEXER : multiplexer port map
             (MODE => MODE,
              LOAD => LOAD(I),
              INPUT => sortieRegistres(I-1),
              OUTPUT => sortieMultiplexers(I));
      end generate MULTIPLEXERX;
      REGX : OneBitRegister port map
        (RESET => RESET,
         CLK => CLK,
         EN => ENABLE,
         D => sortieMultiplexers(I),
         Q => sortieRegistres(I));
   end generate rdc_load;
OUTPUT <= sortieRegistres(N-1);
end Behavioral;
