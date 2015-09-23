<?xml version="1.0" encoding="UTF-8"?>
<xsl:stylesheet id="outlier-html" version="1.0" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:ns="http://www.prime-xs.eu/ms/qcml" xmlns="">
    <xsl:template match="/">
        <html style="font-family:Arial;">
            <style>
                body {
                    width: 1600px;
                    margin: auto;
                    text-align: center;
                }
                figure {
                    width: 500px;
                    margin: 0;
                }
                img {
                    max-height: 100%;
                    max-width: 100%
                }
                table {
                    margin: auto;
                }
                hr {
                    border-color: #000000;
                    border-style: solid;
                }
                .border {
                    border-color: #000000;
                    border-width: 2px;
                    border-style: solid;
                }
                .table-header {
                    background-color: #CCCCCC;
                }
            </style>
            <body>
                <div>
                    <h1>Outlier Detection and Interpretation</h1>
                    <!-- overview -->
                    <p class="border">
                        <h2>Combined Analysis</h2>
                        <xsl:for-each select="ns:qcML/ns:setQuality">
                            <table>
                                    <tr><td colspan="3"><h3>Visualization</h3></td></tr>
                                    <tr>
                                        <td>
                                            <figure>
                                                <img><xsl:attribute name="src">data:image/svg+xml;base64,
                                                    <xsl:value-of select="ns:attachment[@ID='time']/ns:binary"/></xsl:attribute>
                                                </img>
                                                <figcaption><b><xsl:value-of select="ns:attachment[@ID='time']/@name"/></b></figcaption>
                                            </figure>
                                        </td>
                                        <td>
                                            <figure>
                                                <img><xsl:attribute name="src">data:image/svg+xml;base64,
                                                    <xsl:value-of select="ns:attachment[@ID='PCA']/ns:binary"/></xsl:attribute>
                                                </img>
                                                <figcaption><b><xsl:value-of select="ns:attachment[@ID='PCA']/@name"/></b></figcaption>
                                            </figure>
                                        </td>
                                        <td>
                                            <figure>
                                                <img><xsl:attribute name="src">data:image/svg+xml;base64,
                                                    <xsl:value-of select="ns:attachment[@ID='t-SNE']/ns:binary"/></xsl:attribute>
                                                </img>
                                                <figcaption><b><xsl:value-of select="ns:attachment[@ID='t-SNE']/@name"/></b></figcaption>
                                            </figure>
                                        </td>
                                    </tr>
                                    <tr>
                                        <td><h3>Preprocessing</h3></td>
                                        <td><h3><h4><xsl:value-of select="ns:attachment[@ID='OutlierScoreHistogram']/@name"/></h4></h3></td>
                                    </tr>
                                    <tr>
                                        <td>
                                            <h4><xsl:value-of select="ns:attachment[@ID='var']/@name"/></h4>
                                            <p>
                                                <table>
                                                    <tr class="table-header">
                                                        <xsl:call-template name="table-header">
                                                            <xsl:with-param name="list"><xsl:value-of select="ns:attachment[@ID='var']/ns:table/ns:tableColumnTypes"/></xsl:with-param>
                                                        </xsl:call-template>
                                                    </tr>
                                                    <xsl:for-each select="ns:attachment[@ID='var']/ns:table/ns:tableRowValues">
                                                        <tr>
                                                            <xsl:call-template name="table-row">
                                                                <xsl:with-param name="list"><xsl:value-of select="." /></xsl:with-param>
                                                            </xsl:call-template>
                                                        </tr>
                                                    </xsl:for-each>
                                                </table>
                                            </p>
                                            <p>
                                                <xsl:value-of select="ns:qualityParameter[@ID='VarianceThreshold']/@name"/> = <xsl:value-of select="ns:qualityParameter[@ID='VarianceThreshold']/@value"/>
                                            </p>
                                            
                                            <h4><xsl:value-of select="ns:attachment[@ID='corr']/@name"/></h4>
                                            <p>
                                                <table>
                                                    <tr class="table-header">
                                                        <xsl:call-template name="table-header">
                                                            <xsl:with-param name="list"><xsl:value-of select="ns:attachment[@ID='corr']/ns:table/ns:tableColumnTypes"/></xsl:with-param>
                                                        </xsl:call-template>
                                                    </tr>
                                                    <xsl:for-each select="ns:attachment[@ID='corr']/ns:table/ns:tableRowValues">
                                                        <tr>
                                                            <xsl:call-template name="table-row">
                                                                <xsl:with-param name="list"><xsl:value-of select="." /></xsl:with-param>
                                                            </xsl:call-template>
                                                        </tr>
                                                    </xsl:for-each>
                                                </table>
                                            </p>
                                            <p>
                                                <xsl:value-of select="ns:qualityParameter[@ID='CorrelationThreshold']/@name"/> = <xsl:value-of select="format-number(ns:qualityParameter[@ID='CorrelationThreshold']/@value, '00.00%')"/>
                                            </p>
                                            
                                        </td>
                                        <td>
                                            <figure>
                                                <img><xsl:attribute name="src">data:image/svg+xml;base64,
                                                    <xsl:value-of select="ns:attachment[@ID='OutlierScoreHistogram']/ns:binary"/></xsl:attribute>
                                                </img>
                                                <figcaption><b><xsl:value-of select="ns:qualityParameter[@ID='OutlierScoreThreshold']/@name"/> = <xsl:value-of select="format-number(ns:qualityParameter[@ID='OutlierScoreThreshold']/@value, '00.00%')"/></b></figcaption>
                                            </figure>
                                        </td>
                                    </tr>
                                
                            </table>
                        </xsl:for-each>
                    </p>
                    <p class="border">
                        <h2>Individual Outliers</h2>
                        <xsl:for-each select="ns:qcML/ns:runQuality">
                            <h3><xsl:value-of select="@ID"/></h3>
                            <p><b><xsl:value-of select="ns:qualityParameter[@name='Outlier score']/@name"/> = <xsl:value-of select="format-number(ns:qualityParameter[@name='Outlier score']/@value, '00.00%')"/></b></p>
                            <table>
                                <center>
                                    <tr>
                                        <td>
                                            <figure>
                                                <img><xsl:attribute name="src">data:image/svg+xml;base64,
                                                    <xsl:value-of select="ns:attachment[@name='Feature importance']/ns:binary"/></xsl:attribute>
                                                </img>
                                                <figcaption><b><xsl:value-of select="ns:attachment[@name='Feature importance']/@name"/></b></figcaption>
                                            </figure>
                                        </td>
                                        <td>
                                            <figure>
                                                <img><xsl:attribute name="src">data:image/svg+xml;base64,
                                                    <xsl:value-of select="ns:attachment[@name='Explanatory subspace']/ns:binary"/></xsl:attribute>
                                                </img>
                                                <figcaption><b><xsl:value-of select="ns:attachment[@name='Explanatory subspace']/@name"/></b></figcaption>
                                            </figure>
                                        </td>
                                    </tr>
                                </center>
                            </table>
                            <xsl:choose>
                                <xsl:when test="position() != last()"><hr/></xsl:when>
                            </xsl:choose>
                        </xsl:for-each>
                    </p>
                </div>
            </body>
        </html>
    </xsl:template>
    
    <xsl:template name="table-header">
        <xsl:param name="list"/>
        <xsl:variable name="newlist" select="concat(normalize-space($list), ' ')"/>
        <xsl:variable name="first" select="substring-before($newlist, ' ')"/>
        <xsl:variable name="remaining" select="substring-after($newlist, ' ')"/>
        <th>
            <xsl:value-of select="$first"/>
        </th>
        <xsl:if test="$remaining">
            <xsl:call-template name="table-header">
                <xsl:with-param name="list" select="$remaining"/>
            </xsl:call-template>
        </xsl:if>
    </xsl:template>
    
    <xsl:template name="table-row">
        <xsl:param name="list"/>
        <xsl:variable name="newlist" select="concat(normalize-space($list), ' ')"/>
        <xsl:variable name="first" select="substring-before($newlist, ' ')"/>
        <xsl:variable name="remaining" select="substring-after($newlist, ' ')"/>
        <td>
            <xsl:value-of select="$first"/>
        </td>
        <xsl:if test="$remaining">
            <xsl:call-template name="table-row">
                <xsl:with-param name="list" select="$remaining"/>
            </xsl:call-template>
        </xsl:if>
    </xsl:template>
</xsl:stylesheet>